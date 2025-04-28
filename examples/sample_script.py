import sys
import json

def main():
    """Sample script that processes input arguments and returns JSON output"""
    args = sys.argv[1:]
    
    result = {
        "received_args": args,
        "processed": True
    }
    
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())