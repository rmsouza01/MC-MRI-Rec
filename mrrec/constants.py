from pathlib import Path

STANDARD_DATA_PATH = Path.cwd().parent.parent / "Data"
USER_SPECIFIED_DATA_PATH = None
# Specify a custom path to the root folder of the CC359 here if files are not found in ./Data/ folder
DATA_PATH = STANDARD_DATA_PATH if USER_SPECIFIED_DATA_PATH is None else Path(USER_SPECIFIED_DATA_PATH)
