import json
import os


class GeneralSettings:
    base_path: str = ""
    
    def __init__(self) -> None:
        cfg_path = self.__get_cfg_file()
        # read config file from ~/.config
        if os.path.exists(cfg_path):            
            with open(cfg_path, "r") as f:
                # load json and save members to self
                conf = json.load(f)
                for key in conf:
                    setattr(self, key, conf[key])            
        pass
    
    def save(self) -> None:
        # save members to json
        cfg_path = self.__get_cfg_file()
        with open(cfg_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)      
            
    def update_setting(self, key: str, value: str) -> None:
        setattr(self, key, value)
        self.save()
        
    def __get_cfg_file(self) -> str:
        return os.path.join(__file__, "..", "..", "..", "config.json")
        