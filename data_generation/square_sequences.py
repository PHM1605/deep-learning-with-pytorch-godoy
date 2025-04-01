import numpy as np

# generate 128 sets of points, each with 2,3 pr 4 points
def generate_sequences(n=128, variable_len=False, seed=13):
    basic_corners = np.array([[-1,-1],[-1,1],[1,1],[1,-1]])
    np.random.seed(seed)
    bases = np.random.randint(4, size=n) # n numbers of 0,1,2 or 3 
    if variable_len:
        lengths = np.random.randint(3, size=n) + 2 # generate n random numbers, number is 2,3 or 4
    else:
        lengths = [4] * n
    directions = np.random.randint(2,size=n)
    points = [
        basic_corners[[(b+i)%4 for i in range(4)]][slice(None,None,d*2-1)][:l] + np.random.randn(1,2)*0.1
        for b, d, l in zip(bases, directions, lengths)
    ]
    return points, directions 




































