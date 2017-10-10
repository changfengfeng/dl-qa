deep learning qa

## model 1:
q -> conv + max_pooling -> q1
a -> conv + max_pooling -> a1

c = q1 M a1

d = [q1, c, a1]

logtis = d * w + b

## model 2:
q -> bi-lstm + max_pooling -> q1
true_a -> bi-lstm + max_pooling -> true_a1
false_a -> bi-lstm + max_pooling -> false_a1

sim_true = sim(q, true_a1)
sim_false = sim(q, false_a1)

loss = max(0, margin - sim_true + sim_false)

这里的max pooling 其实就在sentence的word中取最大的，在分类时，也可以考虑使用mean

### model1 和 model2 最的区别，就是question和answer是否可以在同一个长度上
如果question和answer长度差别特别大，那就使用model1，如果长度差不多，可以使用model2

## model 3:
bi-lstm + attention
conv net
c = [q1, c, a1]
softmax
