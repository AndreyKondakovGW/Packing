from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.animation import FuncAnimation

fig_colors = {
    'A': 'indigo',
    'B': 'darkviolet',
    'C': 'orangered',
    'D': 'red',
    'F': 'slategray',
    'G': 'darkgoldenrod',
    'H': 'lime',
    'I': 'gold',
    'J': 'black',
    'K': 'palegreen',
}

fig_number = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
}

def fig_box_complete_h(board):
    board_figures = np.unique(np.array(board.board))
    empty_cells = 0
    for fig in board_figures:
        if fig != 0:
            fig_box = np.where(board.board == fig)
            #calculate num empty cells in fig box
            for i in range(len(fig_box[0])):
                if board.board[fig_box[0][i]][fig_box[1][i]] == 0:
                    empty_cells += 1
    return empty_cells

class Board:
    def __init__(self, size):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        self.fig_pos = {}

    def dec_size(self):
        self.size -= 1
        self.board = self.board[:-1]
        for row in self.board:
            row = row[:-1]

    def inc_size(self):
        self.size += 1
        for row in self.board:
            row.append(0)
        self.board.append([0 for _ in range(self.size)])

    def is_free(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == 0
    
    def place(self, figure, pos):
        self.fig_pos[figure.name] = pos
        for tile in figure.tiles:
            self.board[pos[0] + tile[0]][pos[1] + tile[1]] = fig_number[figure.name]
    def move(self, fig, new_pos):
        old_pos = self.fig_pos[fig.name]
        for tile in fig.tiles:
            self.board[old_pos[0] + tile[0]][old_pos[1] + tile[1]] = 0
        for tile in fig.tiles:
            self.board[new_pos[0] + tile[0]][new_pos[1] + tile[1]] = fig_number[fig.name]
        self.fig_pos[fig.name] = new_pos

    def calculate_empty(self):
        empty = 0
        for row in self.board:
            for cell in row:
                if cell == 0:
                    empty += 1
        return empty
    
    def draw(self, use_matplotlib=True, save=False, name='board.png'):
        if not use_matplotlib:
            for row in self.board:
                for cell in row:
                    print(cell, end=' ')
                print()
        else:
            cmap = colors.ListedColormap(['white'] + [fig_colors[fig] for fig in fig_colors])
            bounds = list(range(len(fig_colors) + 2))
            bounds = [x - 0.5 for x in bounds]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            plt.title(f'Packing problem solution. Size: {self.size}')
            plt.pcolor(np.array(self.board), cmap=cmap, norm=norm, edgecolors='k')
            if save:
                plt.savefig(name)
                plt.clf()
    
    def find_best_place(self, figure):
        num_all_try = 0
        for x in range(self.size - figure.size_w + 1):
            for y in range(self.size - figure.size_h + 1):
                placed = figure.try_place(self, (x, y))
                num_all_try += 1
                if placed:
                    return (x, y), num_all_try
        return None, num_all_try

def greedy_search(figures, create_gif=False, gif_name='board.gif'):
    if create_gif:
        all_boards = []
    cost = 0
    min_board_size = int(sqrt(sum([fig.space() for fig in figures]))) 
    #print('Min board size:', min_board_size)
    board = Board(min_board_size)
    for figure in figures:
        pos, num_try = board.find_best_place(figure)
        cost += num_try
        while pos is None:
            board.inc_size()
            pos, num_try = board.find_best_place(figure)
            cost += num_try
        board.place(figure, pos)
        if create_gif:
            all_boards.append(np.array(board.board.copy()))
    #print('Cost:', cost)

    if create_gif:
        fig = plt.figure()
        cmap = colors.ListedColormap(['white'] + [fig_colors[fig] for fig in fig_colors])
        bounds = list(range(len(fig_colors) + 2))
        bounds = [x - 0.5 for x in bounds]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        plt.title(f'Packing problem solution')
        def update(i):
            pcolor_fig = plt.pcolor(np.zeros(all_boards[i].shape), cmap=cmap, norm=norm, edgecolors='k')
            pcolor_fig.set_array(all_boards[i])
            return pcolor_fig,

        anim = FuncAnimation(fig, update, frames=len(all_boards), blit=True)
        anim.save(gif_name, writer='imagemagick', fps=0.5)
    return board

def local_search(figures, max_itter=1000, return_resarch_result=False):
    if not return_resarch_result:
        print("Total number of figures:", len(figures))
        print("Total nuber of permutations of figures:", np.math.factorial(len(figures)))
        current_board = greedy_search(figures, create_gif=True, gif_name='board_iter0.gif')
        print('Initial board size:', current_board.size)
        current_board.draw(save=True, name=f'board_iter0.png')
    else:
        current_board = greedy_search(figures)

    result_x = [0]
    result_y = [current_board.size]
    for i in range(1,max_itter):
        new_fig_list = figures.copy()
        rand_fig_ind1 = np.random.randint(0, len(figures))
        rand_fig_ind2 = np.random.randint(0, len(figures))
        fig1 = new_fig_list[rand_fig_ind1]
        new_fig_list[rand_fig_ind1] = new_fig_list[rand_fig_ind2]
        new_fig_list[rand_fig_ind2] = fig1

        new_board = greedy_search(new_fig_list)
        if new_board.size <= current_board.size:
            result_x.append(i)
            result_y.append(new_board.size)
            if new_board.size < current_board.size:
                if not return_resarch_result:
                    print('Find better order:', [fig.name for fig in new_fig_list])
                    greedy_search(new_fig_list, create_gif=True, gif_name=f'board_iter{i}.gif')
                    new_board.draw(save=True, name=f'board_iter{i}.png')
                    print('Find New board size:', new_board.size)
            figures = new_fig_list
            current_board = new_board
    if not return_resarch_result:
        plt.figure(figsize=(10, 5))
        plt.plot(result_x, result_y)
        plt.title('Local search')
        plt.xlabel('Iteration')
        plt.ylabel('Board size')
        plt.savefig('local_search_result.png')

    return result_x, result_y

            
class Figure:
    def __init__(self,
                 name,
                 tiles):
        self.name = name
        self.tiles = tiles
        self.size_w = max([x[0] for x in tiles]) + 1
        self.size_h = max([x[1] for x in tiles]) + 1

    def try_place(self, board, pos):
        for tile in self.tiles:
            if not board.is_free(pos[0] + tile[0], pos[1] + tile[1]):
                return False
        return True
    
    def space(self):
        return len(self.tiles)
    
if __name__ == '__main__':
    figure = Figure('A', [(0, 0), (0, 1), (1, 1), (1, 2), (2,1), (2,2)])
    figure2 = Figure('B', [(0, 0), (1, 0), (0, 1), (1, 1)])
    figure3 = Figure('C', [(0, 0), (1, 0), (0, 1), (1,1), (2,0), (2,1), (3,0),(3,1), (3,2), (4,0), (4,1), (4,2)])
    figure4 = Figure('D', [(0,0), (1,0), (2,0), (3,0), (4,0), (4,1), (4,2), (4,3), (4,4), (3,2), (3,3), (3,4), (2,3), (2,4), (1,3), (1,4), (0,3), (0,4)])
    figure5 = Figure('F', [(0,6), (0,7), (1,6), (1,7), (2,6), (2,7),(3,0), (3,1),(3,2), (3,3),(3,4), (3,5),(3,6), (3,7),
                           (4,0), (4,1),(4,2), (4,3),(4,4), (4,5),(4,6), (4,7)])
    figure6 = Figure('G', [(0,0), (0,1), (0,2), (0,3),
                           (1,0), (1,1), (1,2), (1,3),
                           (2,0), (2,1), (2,2), (2,3),
                           (3,0), (3,1), (3,2), (3,3),
                           (4,0), (4,1), (4,2), (4,3)])
    figure7 = Figure('H', [(0,0), (0,1), (0,2), (0,3),(0,4), (0,5), (0,6),
                           (1,0), (1,1), (1,2), (1,3),(1,4), (1,5), (1,6),
                           (2,0), (2,1), (2,5), (2,6),
                            (3,5), (3,6),
                           ])
    figure8 = Figure('I', [(0,0), (0,1),
                           (1,0), (1,1),
                           (2,0), (2,1),
                           (3,0), (3,1),(3,2),(3,3),(3,4),(3,5)
                           ])
    figure9 = Figure('J', [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),(0,8),(0,9)])
    figure10 = Figure('K', [(0,0), (0,1), (0,2), (0,3),
                           (1,1), (1,2), (1,3),
                           (2,1), (2,2), (2,3),
                           (3,1), (3,2), (3,3),])
    all_figures = [figure, figure2, figure4, figure3, figure7, figure6, figure5, figure8, figure9, figure10]
    #shuffle figures
    seed = 144
    #144
    np.random.seed(seed)
    all_figures = np.random.permutation(all_figures)
    board = local_search(all_figures, 300)


    number_of_experiments = 200
    seeds = np.random.permutation(range(1000))[:number_of_experiments]
    plt.figure(figsize=(10, 5))
    for seed in seeds:
        np.random.seed(seed)
        all_figures = np.random.permutation(all_figures)
        x,y = local_search(all_figures, 500, return_resarch_result=True)
        plt.plot(x,y)
    plt.show()
    
    