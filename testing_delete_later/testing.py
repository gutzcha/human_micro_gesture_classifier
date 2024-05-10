import argparse
import time
def greet(name, age):
    print(f"Hello, {name}! You are {age} years old.")
    print("Goodbye!")
    # time.sleep(1000)



if __name__ == "__main__":

    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification',
                                     add_help=False)
    parser.add_argument('--name', type=str, help='Your name')
    parser.add_argument('--age', type=int, help='Your age')

    args = parser.parse_args()
    greet(args.name, args.age)
