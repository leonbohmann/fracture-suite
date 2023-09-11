import os



def find_file(path: str, file: str) -> str | None:
# find .pkl file in scalppath
    if not os.path.exists(path):
        return None

    for file in os.listdir(path):
        if file.endswith(".pkl"):
            return os.path.join(path, file)
        
    return None