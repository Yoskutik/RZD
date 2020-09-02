import pandas as pd
import numpy as np
import time


start_time = time.time()
df = pd.read_csv('data/df_renamed.csv', index_col=0)
df_prices = pd.read_csv('data/prices.csv', index_col=0)
print(f"--- {(time.time() - start_time):.3f} seconds ---")

graph = {}
for c in df.frm.unique():
    graph[c] = list(filter(lambda x: x in df.frm.values, df[df.frm == c].to.unique()))


def get_cost(path):
    price = 0
    for i in range(len(path) - 1):
        price += df_prices.loc[f'{path[i]} -> {path[i + 1]}', 'Price']
    return price


def find_best_path(start, end, max_len=5):
    assert start in np.unique(df.values), f'{start} is not presented in the dataset'
    assert end in np.unique(df.values), f'{end} is not presented in the dataset'

    fringe = [(start, [start])]
    best_path = None
    best_cost = np.Inf
    while fringe:
        state, path = fringe.pop()
        if state == end:
            cost = get_cost(path)
            if cost < best_cost:
                best_cost = cost
                best_path = path
            continue
        if len(path) >= max_len:
            continue
        if get_cost(path) > best_cost:
            continue
        for next_state in graph[state]:
            if next_state in path and next_state != end:
                continue
            fringe.append((next_state, path+[next_state]))
    return best_path, best_cost


def find_final_path(nodes):
    total_cost = 0
    for i in range(len(nodes) - 1):
        path, cost = find_best_path(nodes[i], nodes[i + 1])
        total_cost += cost
        print(' -> '.join(path))
    print(f'Минимальная стоимость пути: {total_cost:.3f}')


start_time = time.time()
find_final_path(['Узел: 0', 'Узел: 2', 'Узел: 1', 'Узел: 3', 'Узел: 5', 'Узел: 0', 'Узел: 20', 'Узел: 21', 'Узел: 17'])
print(f"--- {(time.time() - start_time):.3f} seconds ---")

