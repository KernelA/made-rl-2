import json
from tabulate import tabulate
from IPython.display import HTML, display

import pandas as pd

HEADER_MAPPING = {"cross_fraction_win": "Доля побед крестиков",
                  "circle_fraction_win": "Доля побед ноликов", "draw_fraction": "Доля ничьих", "players": "Стратегии игроков"}


def load_json(path_to_file: str):
    with open(path_to_file, "r") as json_file:
        game_stat = json.load(json_file)

    return game_stat


def show_tables(stat_dir):
    rows = []
    header = ["Вероятность совершить случайное действие"]
    for file in stat_dir.glob("**/*.json"):
        proba = file.parent.name.split("_")[1]
        game_stat = load_json(str(file))
        rows.append((proba,) + tuple(map(str, game_stat.values())))
        if len(header) == 1:
            header.extend(list(map(lambda x: HEADER_MAPPING[x], game_stat.keys())))

    return display(HTML(tabulate(rows, headers=header, tablefmt="html")))


def game_stat_to_table(stat_dir):
    rows = []
    for file_path in stat_dir.rglob("game_stat.json"):
        game_stat = load_json(str(file_path))
        players = file_path.parent.name
        game_stat["players"] = players
        rows.append(game_stat)

    data = pd.DataFrame.from_records(rows)
    other_columns = data.columns.drop("players").to_list()
    data = data[["players"] + other_columns]
    data.sort_values("players", inplace=True)
    data.rename(HEADER_MAPPING, axis="columns", inplace=True)

    return data
