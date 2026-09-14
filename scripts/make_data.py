from data_collection import DataCollector
import os

root = os.getcwd()
data_folder = os.path.join(root, "datasets")

size = 50 # change only the size for an entirely new data corpus.
folder_name = f"data_{size}_MB"
folder_path = os.path.join(data_folder, folder_name)

os.makedirs(folder_path, exist_ok=True)

output_path = os.path.join(folder_path, "output.txt")
report_path = os.path.join(folder_path, "report.txt")

"""
CORPUS will follow a 60:40 ratio in favour of languages.
6 languages, X / 10 MB. -> TOTAL is 3X/5
5 programming languages, X / 12.5 MB. -> TOTAL is 2X/5.

RANDOM SEED WILL ALWAYS BE KEPT 42 for the official data corpus. 
"""

dataObj = DataCollector(output_path, report_path)
dataObj.add_language("english", (size // 10), 42)
dataObj.add_language("hindi", (size // 10), 42)
dataObj.add_language("spanish", (size // 10), 42)
dataObj.add_language("german", (size // 10), 42)
dataObj.add_language("french", (size // 10), 42)
dataObj.add_language("japanese", (size // 10), 42)

dataObj.add_programming_language("python", (size // 12.5), 42)
dataObj.add_programming_language("cpp", (size // 12.5), 42)
dataObj.add_programming_language("javascript", (size // 12.5), 42)
dataObj.add_programming_language("java", (size // 12.5), 42)
dataObj.add_programming_language("rust", (size // 12.5), 42)

"""
I CANNOT BELIEVE THE AMOUNT OF MODULARISATION I HAVE ACHIEVED.
MODULARISATION IS PEAK. IT JUST CLICKS.
"""


