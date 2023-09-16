import os
from typing import Annotated

import numpy as np
import typer
from apread import APReader
from rich import print
from rich.progress import track

from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import find_file
from fracsuite.tools.specimen import fetch_specimens

app = typer.Typer()
general = GeneralSettings()


def reader_to_csv(reader: APReader, out_dir, dot: str = "."):
    """Writes the reader data to a csv file."""
    # create csv file
    csv_file = os.path.join(out_dir, f"{reader.fileName}.csv")
    with open(csv_file, 'w') as f:
        # write header
        f.write(";")
        for chan in reader.Channels:
            f.write(f"{chan.Name} [{chan.unit}];")
        f.write("\n")
        
        # write header
        f.write("Maximum;")
        for chan in reader.Channels:
            f.write(f"{np.max(chan.data)};")
        f.write("\n")
        
        f.write("Minimum;")
        for chan in reader.Channels:
            f.write(f"{np.min(chan.data)};")
        f.write("\n")
        
        f.write("Time of Maximum;")
        for chan in reader.Channels:
            max_i = np.argmax(chan.data)
            if not chan.isTime:
                time = chan.Time.data[max_i]
                f.write(f"{time};")
            else:
                f.write(";")
        f.write("\n")
        
        # write data
        max_len = np.max([len(x.data) for x in reader.Channels])
        for i in track(range(0, max_len)):
            f.write(";")
            for g in reader.Groups:       
                if i < len(g.ChannelX.data):
                    f.write(f"{g.ChannelX.data[i]};")         
                for chan in g.ChannelsY:
                    if i < len(chan.data):
                        f.write(f"{chan.data[i]};")
                    else:
                        f.write(";")
            f.write("\n")

    with open(csv_file, 'r') as f:
        content = f.read()
        content = content.replace(".", dot)
    with open(csv_file, 'w') as f:
        f.write(content)
        
        
@app.command()
def to_csv(
    specimen_name: Annotated[str, typer.Argument(help="The name of the specimen to convert.")],
    number_dot: Annotated[str, typer.Option(help="Number format dot.")] = ".",
    plot: Annotated[bool, typer.Option(help="Plot the reader before saving.")] = False):
    """Converts the given specimen to a csv file."""
    specimen = fetch_specimens(specimen_name, general.base_path)[0]

    acc_path = os.path.join(specimen.path, "fracture", "acceleration")
    acc_file = find_file(acc_path, "*.BIN")

    if acc_file is None:
        print(f"Could not find acceleration file for specimen '{specimen_name}'.")
        return
    
    reader = APReader(acc_file)
    
    if plot:
        reader.plot()    
        
    reader_to_csv(reader, acc_path, number_dot)
    
