# UCC AI-Quest

-Replaced the Resnet50 with the Densenet161.
Reason for this is to experiment if choosing the Densenet would maybe improve overall accuracy, and also on the other hand, for personal entertainment.

They both have their advantages and such, both tackling the vanishing gradient problem differently:
  -Resnet introduces the concept of residual learning, where the use of skip connections are implemented. So instead of learning the desired underlying mapping directly, it learns the residual mapping, the difference between the input and output of that particular layer. Overall the use of skip connections allows "shortcuts" that bypasses one or more layers. These connections take the input from an earlier layer and adds it to the output of a later layer. 

  -Densenet on the other hand focus its dense connectivity when addressing this isssue. Each layer receives an input from all previous layers in the block. This facilitates the flow of the gradient during backpropagation. Simpler terms, it provides multiple paths for the gradient to flow. Another thing I like about using Densenet is the concept of parameter sharing, which means the network becomes more compact and efficient when it comes to the number of parameters needed for it to learn.

So far its still in testing.
 refer back to this link if stuck https://github.com/ReML-AI/ucc-ai-quest-baseline
