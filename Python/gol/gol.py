

from itertools import product
from collections.abc import Iterator
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


# I grabbed these from somewhere...will update with link
seeds = {
    "glider": [[0,1,0],[0,0,1],[1,1,1]],
    "diehard": [
        [0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 1],
    ],
    "boat": [[1, 1, 0], [1, 0, 1], [0, 1, 0]],
    "r_pentomino": [[0, 1, 1], [1, 1, 0], [0, 1, 0]],
    "pentadecathlon": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "beacon": [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
    "acorn": [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 1]],
    "spaceship": [[0, 0, 1, 1, 0], [1, 1, 0, 1, 1], [1, 1, 1, 1, 0], [0, 1, 1, 0, 0]],
    "block_switch_engine": [
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
    ],
    "infinite": [
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 0, 1],
    ],
}


    
def rand_board(size:int) -> NDArray:
    '''Generates a single size x size randomly populated board'''
    board = np.random.randint(0,2,(size,size),dtype='int8')
    return board



def seed_board(size:int, seed_name:str, pos:tuple[int,int]) -> NDArray:
    '''
    Generates a size x size board with specified seed located at pos.
    '''
    board = np.zeros((size,size),dtype='int')    
    seed = np.asarray(seeds[seed_name])    
    board[ pos[0]:pos[0]+seed.shape[0], pos[1]:pos[1]+seed.shape[1] ] = seed    
    return board
    


def multi_step(board: NDArray, iters: int) -> NDArray:
    
    '''Advance board by iters iterations'''
    
    for _ in range(iters):
        board = get_next(board)
    return board


#This one-liner GOL borrowed from here:
#https://www.reddit.com/r/Python/comments/qyaa9t/explaining_single_line_conways_game_of_life_from/

def get_next(b: NDArray) -> NDArray:
    
    '''Advance board b by single step. Note that edges wrap!'''
    
    b=(3==(n:=sum(np.roll(b,s,(0,1))for s in product(*2*[(-1,0,1)]))))|(n==4)&b        
    return b

def board_generator(size: int, iters:int) -> Iterator[tuple[NDArray, NDArray]]:
    
    '''Returns an initial random board and the corresponding board 
    advanced by iters iterations'''
    
    while True:
        board_in = rand_board(size)    
        board_out = multi_step(board_in, iters)
        yield board_in, board_out
            
        
def animate(board: NDArray, iters: int|None) -> None:
    '''
    Displays a figure animating the input board through iters iterations, or
    indefinitely if iters is None.
    '''
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    image = ax.imshow(board)
    plt.draw()
    valid = True
    count = 0
    while valid:
        plt.pause(0.10) 
        board = get_next(board)
        image.set_data(board)
        plt.draw()
        count += 1
        if iters and (count==iters):
            valid = False


# this is not the most efficient way to do this...
def make_dataset(size:int, iters:int, nboards:int, desc='') -> tuple[NDArray,NDArray]:
    
    ''' Create a dataset of input and output boards. 
    
    size -> board size (will be size x size)
    iters -> number steps to simulate (only 1st and last pairs are used)
    nboards -> total number of examples
    desc -> optional user description for saved file names
    '''
    
    inputs = []
    outputs = []
    board_iter = board_generator(size, iters)
    for _ in tqdm(range(nboards)):
       (bi, bo) = next(board_iter)
       inputs.append(bi)
       outputs.append(bo)
   
    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)

    np.save(f'gol_inputs_{size}_{iters}_{desc}', inputs)
    np.save(f'gol_outputs_{size}_{iters}_{desc}', outputs)

    return inputs, outputs


if __name__ == "__main__":
    
    board = seed_board(8,'glider',(1,1))
    animate(board, 50)
    
    # optinally generate a dataset
    #size = 8
    #iters = 1
    #nboards = 1000
    
    #make_dataset(size,iters,nboards,'test')
    



    
    
    





