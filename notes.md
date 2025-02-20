## Algorithm: Multiclass Perceptron Learning Algorithm


### perceptron = helper

### step 1: forward pass = look at picture
- perception has 784 weights, one for each pixel and a bias (number)
- take images 784 pixel values, multiply by matching weight, add them all up, then add bias
- like each pixel a vote on whether its their digit, then tallying the votes
  - Score = (w1 * p1) + (w2 * p2) + ... + bias
- each of the 10 perceptions does this, the one with the highest score says its their digit


### step 2: check if perceptron with highest score is right or wrong
- from the labels, we know the real answer 
- if the perceptron with the highest score is not the correct one
  - for each perceptron:
    - take perceptron with highest score
      - if perception is correct, matching label, then do nothing
      - if perceptron is incorrect, not matching label, then each perceptron is tweaked based off how wrong they are
        - if perceptron should have won, then buff
        - if perceptron shouldn't have won, then demote
        - if perceptron didn't win or shouldn't have won, then do nothing
          - i.e their scores are lower than the correct perceptron score
      - if perceptron did win, and should have won, do nothing


### step 3: Fix their mistakes (back propagation)
- each perceptron calculates, if wrong, how wrong its weights were, and adjusts
- its adjust = (tweak = how wrong perceptron was (+1 or -1)) * (pixel value)
- bias also tweaked based off how wrong perceptron was (+1 or -1)

### step 4: Learn a little bit (gradient descent)
- perceptrons take small steps so to no overreact
- the steps are set by the learning rate
  - new weight = old weight + (learning rate * tweak), tweak = (how wrong(+1 or -1)) * (pixel value)

### step 5: Learn in groups (batches)
- instead of a single picture at a time, we do 100 pictures at once
- each perceptron adds up all the tweaks from those 100 pictures, averages them, then updates their weights
- makes learning faster and smoother

### step 6: keep practicing 
- show all 60,000 pictures once, thats one 'epoch', perceptrons tweak their weights a little each batch, getting better

