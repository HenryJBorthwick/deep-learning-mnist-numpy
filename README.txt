# Answer the conceptual questions here
Q1: Is there anything we need to know to get your code to work? If you did not get your code working or handed in an incomplete solution please let us know what you did complete (0-4 sentences)
- i have changed file paths to make it work on locally on machine
- may need to update paths back to default to get to work
- my code achieves test accuracy over 80%
- i completed the preprocess, model, 


Q2: Why do we normalize our pixel values between 0-1? (1-3 sentences)
- faster convergence as it standardizes the input data or pixel values on to the same scale, this in turn prevents the model from taking longer to adjust weights due to varying scales, meaning efficient training
- avoid numeric issues from large pixel values (i.e 255) during matrix operations, risking overflow or underflow in floating point computations


Q3: Why do we use a bias vector in our forward pass? (1-3 sentences)
- shifts decision boundaries in a perceptron, lets the model adjust independent of the input data
- improves the classification as without the bias, boundary must pass through 0,0 where MNIST aren't necessarily centered at zero
- gives expressive power which enables the model to capture complex patterns

Q4: Why do we separate the functions for the gradient descent update from the calculation of the gradient in back propagation? (2-4 sentences)
- code organization make code cleaner and easier to follow and isolate problems if something goes wrong 
- flexibility with the ability to modify the optimization (learning rate) strategy without changing gradient logic


Q5: What are some qualities of MNIST that make it a “good” dataset for a classification problem? (2-3 sentences)
- size with 60,000 training and 10,000 test images, offers enough data for both learning and evaluation
- easy to use with the greyscale uniform sized and centered images simplifying preprocessing avoiding resizing or color handling
- realistic data with a variety of hand writing without extreme distortions


Q6: Suppose you are an administrator of the NZ Health Service (CDHB or similar). What positive and/or negative effects would result from deploying an MNIST-trained neural network to recognize numerical codes on forms that are completed by hand by a patient when arriving for a health service appointment? (2-4 sentences)
- positive effects:
- automate recognition of hand written numeric codes
- speeding up patient check in
- reducing administrative work, 
- minimizing human entry errors

- negative effects:
- may misclassify numbers due to handwriting styles not captures in the MNIST data set 
- delay patient care 
- could raise privacy and regulatory concerns if patient data is processed without proper safeguards and practices
