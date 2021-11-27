import json
from tabulate import tabulate
from IPython.display import HTML, display


def show_tables(stat_dir):
    rows = []
    map_header = {"cross_fraction_win": "Доля побед крестиков",
                  "circle_fraction_win": "Доля побед ноликов", "draw_fraction": "Доля ничьих"}
    header = ["Вероятность совершить случайное действие"]
    for file in stat_dir.glob("**/*.json"):
        proba = file.parent.name.split("_")[1]
        with open(file, "r") as json_file:
            game_stat = json.load(json_file)
        rows.append((proba,) + tuple(map(str, game_stat.values())))
        if len(header) == 1:
            header.extend(list(map(lambda x: map_header[x], game_stat.keys())))

    return display(HTML(tabulate(rows, headers=header, tablefmt="html")))
