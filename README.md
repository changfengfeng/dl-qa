deep learning qa

model 1:
q -> conv + max_pooling -> q1
a -> conv + max_pooling -> a1

c = q1 M a1

d = [q1, c, a1]

logtis = d * w + b

mode 2:
q -> bi-lstm + conv + max_pooling -> q1
true_a -> bi-lstm + conv + max_pooling -> true_a1
false_a -> bi-lstm + conv + max_pooling -> false_a1

sim_true = sim(q, true_a1)
sim_false = sim(q, false_a1)

loss = max(0, margin - sim_true + sim_false)

