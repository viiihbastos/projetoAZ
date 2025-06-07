from databricks.solutions import Accelerator
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="databricks path")
parser.add_argument("-n", "--name", help="solution codename")
parser.add_argument("-m", "--markdown", help="split solution by markdown sections", action='store_true')

args = parser.parse_args()

if not os.environ['DB_HOST']:
    print("Please specify DB_HOST environment variable")
    exit(1)

if not os.environ['DB_TOKEN']:
    print("Please specify DB_TOKEN environment variable")
    exit(1)

if not args.name:
    print("please provide a code name for solution accelerator")
    exit(1)

if not args.path:
    print("please provide a databricks path to download solution from")
    exit(1)

Accelerator(
    db_host=os.environ['DB_HOST'],
    db_token=os.environ['DB_TOKEN'],
    db_path=args.path,
    db_name=args.name,
    markdown=args.markdown
).release()
