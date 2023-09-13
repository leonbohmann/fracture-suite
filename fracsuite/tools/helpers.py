import os



def find_file(path: os.PathLike, ext: str) -> str | None:
           
    if not os.path.exists(path):
        return None

    for file in os.listdir(path):
        if file.endswith(ext):
            return os.path.join(path, file)
        
    return None