import os

file_path = os.path.join(r'D:\Project-mpg microgesture\mac2024\track1\annotations-20240613T185539Z-001\annotations','label_name.txt')
# file_path = os.path.join(r'D:\Project-mpg microgesture\mac2024\track1\annotations-20240613T185539Z-001\annotations','fine2coarse.txt')

# Read the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Create the dictionary
gesture_dict = {}
for line in lines:
    index, description = line.split(maxsplit=1)
    gesture_dict[int(index)] = description.strip()

# Print the dictionary to verify
# print(gesture_dict)
for key, value in gesture_dict.items():
    print(f'{key}: "{value}",')