# Enregistre les logs dans un fichier prediction.log à la racine du projet
import datetime
import json
from pathlib import Path


class Logger():

    def __init__(self):
        pass

    def log(self, input, output, duration):
        with open("prediction.log", "a") as f:
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            json.dump({"date": date,"input": input, "output": output, "duration": duration}, f)
            f.write("\n")

    def get_logs(self):

        base_dir = Path(__file__).resolve().parent.parent

        with open(base_dir / "prediction.log", "r") as f:
            return [json.loads(line) for line in f]