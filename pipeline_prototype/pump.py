import argparse
from pipeline_prototype.redis_q import RedisQueue


def cli_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param", required=True,
                        help="Parameter to pass into redis")
    parser.add_argument("--redishost", default="127.0.0.1",
                        help="Address at which the redis server can be reached")
    parser.add_argument("--redisport", default="6379",
                        help="Port on which redis is available")
    parser.add_argument("-q", "--qname", default="input",
                        help='Name of the redis-queue to send param to. Default: input')
    args = parser.parse_args()

    redis_queue = RedisQueue(args.redishost, port=int(
        args.redisport), prefix=args.qname)
    redis_queue.connect()

    if args.param == 'None':
        redis_queue.put(None)
    else:
        redis_queue.put(int(args.param))
    print(f"Pushed {args.param} to Redis Queue:{args.qname}")


if __name__ == "__main__":
    cli_run()
