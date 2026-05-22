/*
 * 百万并发TCP服务端 —— Linux epoll ET模式
 *
 * 编译: gcc -O2 -Wall -o tcp_server tcp_server.c
 * 运行前系统调优:
 *   ulimit -n 1048576
 *   sysctl -w net.core.somaxconn=65535
 *   sysctl -w net.ipv4.tcp_max_syn_backlog=65535
 *   sysctl -w net.core.netdev_max_backlog=65535
 *   sysctl -w net.ipv4.ip_local_port_range="1024 65535"
 *   sysctl -w net.ipv4.tcp_tw_reuse=1
 *   sysctl -w net.ipv4.tcp_fin_timeout=10
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

/* ========== 可调参数 ========== */
#define MAX_EVENTS   65536    /* epoll_wait 单次返回最大事件数 */
#define PORT         8080     /* 监听端口 */
#define LISTEN_BACKLOG 65535  /* listen backlog */
#define BUFFER_SIZE  4096     /* 读写缓冲区 */
#define MAX_CONN     1048576  /* 最大连接数(通过 -m 参数可覆盖) */
#define STATS_INTERVAL 10     /* 每N秒打印一次统计 */

/* ========== 连接状态 ========== */
typedef struct {
    int      fd;         /* socket fd */
    uint32_t events;     /* 当前监听的事件 */
    time_t   last_active;
    char     ip[INET6_ADDRSTRLEN];
    uint16_t port;
    uint32_t padding;    /* 对齐到 64 字节 */
} conn_t;

/* ========== 全局变量 ========== */
static conn_t         *g_conns = NULL;     /* 连接数组（直接索引用fd，简单高效） */
static int             g_conns_cap = 0;    /* 数组容量 */
static volatile int    g_running = 1;
static int             g_epfd;

/* 统计 */
static volatile uint64_t g_total_accept  = 0;
static volatile uint64_t g_total_close   = 0;
static volatile uint64_t g_curr_conns    = 0;
static volatile uint64_t g_bytes_read    = 0;
static volatile uint64_t g_bytes_write   = 0;

/* ========== 工具函数 ========== */

/* 设置非阻塞 */
static int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

/* 扩展连接数组 */
static int expand_conns(int new_cap) {
    conn_t *p = realloc(g_conns, (size_t)new_cap * sizeof(conn_t));
    if (!p) return -1;
    g_conns = p;
    /* 初始化新增部分 */
    for (int i = g_conns_cap; i < new_cap; i++) {
        g_conns[i].fd = -1;
    }
    g_conns_cap = new_cap;
    return 0;
}

/* 获取当前时间戳(毫秒) */
static uint64_t now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

/* ========== 连接管理 ========== */

/* 注册新连接 */
static conn_t* conn_add(int fd, struct sockaddr_in *addr) {
    if (fd >= g_conns_cap) {
        int new_cap = fd + 65536;
        if (new_cap > MAX_CONN + 65536) new_cap = MAX_CONN + 65536;
        if (expand_conns(new_cap) != 0) return NULL;
    }

    conn_t *c = &g_conns[fd];
    c->fd = fd;
    c->events = EPOLLIN | EPOLLET; /* 边缘触发 + 只监听读 */
    c->last_active = time(NULL);
    c->port = ntohs(addr->sin_port);
    inet_ntop(AF_INET, &addr->sin_addr, c->ip, sizeof(c->ip));

    __atomic_add_fetch(&g_total_accept, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_curr_conns, 1, __ATOMIC_RELAXED);

    struct epoll_event ev = {
        .events   = c->events,
        .data.ptr = c
    };
    epoll_ctl(g_epfd, EPOLL_CTL_ADD, fd, &ev);
    return c;
}

/* 移除连接 */
static void conn_del(conn_t *c) {
    if (!c || c->fd < 0) return;
    epoll_ctl(g_epfd, EPOLL_CTL_DEL, c->fd, NULL);
    close(c->fd);
    c->fd = -1;

    __atomic_add_fetch(&g_total_close, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&g_curr_conns, 1, __ATOMIC_RELAXED);
}

/* 修改监听事件(ET模式需要重新ARM) */
static void conn_mod(conn_t *c, uint32_t events) {
    c->events = events;
    struct epoll_event ev = {
        .events   = events,
        .data.ptr = c
    };
    epoll_ctl(g_epfd, EPOLL_CTL_MOD, c->fd, &ev);
}

/* ========== 网络读写 ========== */

/* ET 模式：循环读直到 EAGAIN */
static int do_read(conn_t *c) {
    char buf[BUFFER_SIZE];
    int total = 0;

    while (1) {
        ssize_t n = read(c->fd, buf, sizeof(buf));
        if (n > 0) {
            total += n;
            __atomic_add_fetch(&g_bytes_read, n, __ATOMIC_RELAXED);
        } else if (n == 0) {
            conn_del(c);
            return -1;
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            conn_del(c);
            return -1;
        }
    }

    c->last_active = time(NULL);

    /* 读到数据后，注册写事件准备回显 */
    if (total > 0) {
        conn_mod(c, EPOLLIN | EPOLLOUT | EPOLLET);
    }
    return total;
}

/* ET模式：循环写直到EAGAIN或无数据 */
static int do_write(conn_t *c) {
    /* 简单的 echo 回显 */
    const char *resp = "HTTP/1.1 200 OK\r\n"
                       "Content-Type: text/plain\r\n"
                       "Content-Length: 13\r\n"
                       "Connection: keep-alive\r\n"
                       "\r\n"
                       "Hello, World!\n";
    size_t len = strlen(resp);
    static __thread size_t offset = 0; /* 每个连接独立偏移量 */

    while (offset < len) {
        ssize_t n = write(c->fd, resp + offset, len - offset);
        if (n > 0) {
            offset += n;
            __atomic_add_fetch(&g_bytes_write, n, __ATOMIC_RELAXED);
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            conn_del(c);
            return -1;
        }
    }

    if (offset >= len) {
        offset = 0;
        /* 写完后恢复只监听读 */
        conn_mod(c, EPOLLIN | EPOLLET);
    }
    return 0;
}

/*
 * 正确的 ET 写：每个连接维护自己的写缓冲
 * 上面的 do_write 是简化版，真正的百万并发需要每连接一个输出缓冲区。
 * 这里提供一个生产级的实现框架：
 */
#if 0
/* ---- 生产级实现参考 ---- */
typedef struct {
    conn_t base;
    char   rbuf[4096];        /* 读缓冲 */
    char  *wbuf;              /* 写缓冲(动态分配) */
    size_t wbuf_size;
    size_t wbuf_offset;
    /* ... 应用层协议状态 ... */
} app_conn_t;

/* 每个连接的写操作 */
static int do_write_proper(app_conn_t *ac) {
    while (ac->wbuf_offset < ac->wbuf_size) {
        ssize_t n = write(ac->base.fd,
                          ac->wbuf + ac->wbuf_offset,
                          ac->wbuf_size - ac->wbuf_offset);
        if (n > 0) {
            ac->wbuf_offset += n;
        } else if (errno == EAGAIN) {
            return 0; /* 等下次 EPOLLOUT */
        } else {
            return -1;
        }
    }
    /* 写完了，切回读 */
    free(ac->wbuf);
    ac->wbuf = NULL;
    conn_mod(&ac->base, EPOLLIN | EPOLLET);
    return 0;
}
#endif

/* ========== 监听socket ========== */

static int create_listen_socket(int port) {
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (fd < 0) {
        perror("socket");
        return -1;
    }

    /* SO_REUSEADDR: 快速重启 */
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

    /* TCP_NODELAY: 禁用Nagle算法 */
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    /* 增大收发缓冲区 */
    int bufsize = 256 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));

    struct sockaddr_in addr = {
        .sin_family      = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port        = htons(port)
    };

    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(fd);
        return -1;
    }

    if (listen(fd, LISTEN_BACKLOG) < 0) {
        perror("listen");
        close(fd);
        return -1;
    }

    printf("[INFO] 监听端口 %d, backlog=%d\n", port, LISTEN_BACKLOG);
    return fd;
}

/* ========== 主循环 ========== */

static void event_loop(int listen_fd) {
    g_epfd = epoll_create1(0);
    if (g_epfd < 0) {
        perror("epoll_create1");
        return;
    }

    /* 注册监听socket(也用ET) */
    struct epoll_event ev = {
        .events   = EPOLLIN | EPOLLET,
        .data.ptr = NULL  /* NULL 表示是 listen fd */
    };
    epoll_ctl(g_epfd, EPOLL_CTL_ADD, listen_fd, &ev);

    struct epoll_event events[MAX_EVENTS];
    time_t last_stats = time(NULL);

    printf("[INFO] 进入事件循环, epfd=%d\n", g_epfd);

    while (g_running) {
        int nfds = epoll_wait(g_epfd, events, MAX_EVENTS, 1000);
        if (nfds < 0) {
            if (errno == EINTR) continue;
            perror("epoll_wait");
            break;
        }

        for (int i = 0; i < nfds; i++) {
            conn_t *c = (conn_t*)events[i].data.ptr;
            uint32_t revents = events[i].events;

            /* listen fd */
            if (c == NULL) {
                while (1) {
                    struct sockaddr_in client_addr;
                    socklen_t clen = sizeof(client_addr);
                    int cfd = accept4(listen_fd,
                                      (struct sockaddr*)&client_addr,
                                      &clen,
                                      SOCK_NONBLOCK);
                    if (cfd < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK)
                            break;
                        perror("accept4");
                        break;
                    }

                    /* 设置TCP选项 */
                    int opt = 1;
                    setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
                    setsockopt(cfd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt));

                    conn_add(cfd, &client_addr);
                }
                continue;
            }

            /* 错误/挂起 */
            if (revents & (EPOLLERR | EPOLLHUP)) {
                conn_del(c);
                continue;
            }

            /* 可读 */
            if (revents & EPOLLIN) {
                if (do_read(c) < 0) continue;
            }

            /* 可写 */
            if (revents & EPOLLOUT) {
                do_write(c);
            }
        }

        /* 定期打印统计 */
        time_t now = time(NULL);
        if (now - last_stats >= STATS_INTERVAL) {
            printf("[STATS] curr=%lu | accepted=%lu | closed=%lu | "
                   "read=%lu MB | write=%lu MB\n",
                   g_curr_conns, g_total_accept, g_total_close,
                   g_bytes_read / (1024 * 1024),
                   g_bytes_write / (1024 * 1024));
            last_stats = now;
        }
    }

    printf("[INFO] 事件循环退出\n");
}

/* ========== 信号处理 ========== */
static void sig_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        printf("\n[INFO] 收到信号 %d, 优雅退出...\n", sig);
        g_running = 0;
    }
}

/* ========== 打印系统配置提示 ========== */
static void print_tuning_guide(void) {
    printf(
        "╔═══════════════════════════════════════════════════════╗\n"
        "║     百万并发TCP服务端 — 系统调优指南                      ║\n"
        "╠═══════════════════════════════════════════════════════╣\n"
        "║  1. 文件描述符限制:                                      ║\n"
        "║     ulimit -n 1048576                                  ║\n"
        "║                                                        ║\n"
        "║  2. 内核参数(/etc/sysctl.conf):                         ║\n"
        "║     net.core.somaxconn = 65535                         ║\n"
        "║     net.ipv4.tcp_max_syn_backlog = 65535               ║\n"
        "║     net.core.netdev_max_backlog = 65535                ║\n"
        "║     net.ipv4.ip_local_port_range = 1024 65535          ║\n"
        "║     net.ipv4.tcp_tw_reuse = 1                          ║\n"
        "║     net.ipv4.tcp_fin_timeout = 10                      ║\n"
        "║     net.ipv4.tcp_rmem = 4096 87380 16777216           ║\n"
        "║     net.ipv4.tcp_wmem = 4096 65536 16777216            ║\n"
        "║                                                        ║\n"
        "║  3. 内存估算(每连接 ~10KB):                             ║\n"
        "║     100万连接 ≈ 10GB 内存(含内核缓冲区)                  ║\n"
        "║                                                        ║\n"
        "║  4. 多IP/多端口: 突破64K端口限制                         ║\n"
        "║     (客户端连接需更多源端口)                              ║\n"
        "╚═══════════════════════════════════════════════════════╝\n\n"
    );
}

/* ========== main ========== */
int main(int argc, char *argv[]) {
    int port = PORT;
    int max_conn = MAX_CONN;

    /* 参数解析 */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            max_conn = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            printf("用法: %s [-p port] [-m max_connections]\n", argv[0]);
            printf("  默认: port=%d, max_conn=%d\n", PORT, MAX_CONN);
            return 0;
        }
    }

    print_tuning_guide();

    /* 安装信号处理 */
    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGPIPE, SIG_IGN);  /* 忽略 SIGPIPE */

    /* 初始化连接数组 */
    if (expand_conns(65536) != 0) {
        fprintf(stderr, "[FATAL] 内存分配失败\n");
        return 1;
    }

    /* 检查 fd 限制 */
    struct rlimit rlim;
    if (getrlimit(RLIMIT_NOFILE, &rlim) == 0) {
        printf("[INFO] 当前文件描述符限制: soft=%lu, hard=%lu\n",
               rlim.rlim_cur, rlim.rlim_max);
        if ((long)rlim.rlim_cur < max_conn + 100) {
            printf("[WARN] fd限制(%lu) < 目标连接数(%d), 请执行 ulimit -n %d\n",
                   rlim.rlim_cur, max_conn, max_conn + 1000);
        }
    }

    /* 创建监听socket */
    int listen_fd = create_listen_socket(port);
    if (listen_fd < 0) return 1;

    /* 主循环 */
    event_loop(listen_fd);

    /* 清理 */
    close(listen_fd);
    close(g_epfd);

    /* 关闭所有残留连接 */
    for (int i = 0; i < g_conns_cap; i++) {
        if (g_conns[i].fd >= 0) {
            close(g_conns[i].fd);
        }
    }
    free(g_conns);

    printf("[INFO] 服务端已退出\n");
    return 0;
}
