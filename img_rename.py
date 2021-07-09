import os
import argparse

def main(folder_name):

    for count, filename in enumerate(os.listdir(folder_name)):
        _, ext = os.path.splitext(filename)
        dst = f"img_{count:03}{ext}"
        src = os.path.join(folder_name, filename)
        dst = os.path.join(folder_name, dst)

        os.rename(src, dst)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, help="Path of the folder")
    args = parser.parse_args()
    main(args.folder_name)
