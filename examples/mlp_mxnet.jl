using MXNet

mlp = @mx.chain mx.Variable(:data)             =>
  mx.FullyConnected(name=:fc1, num_hidden=128) =>
  mx.Activation(name=:relu1, act_type=:relu)   =>
  mx.FullyConnected(name=:fc2, num_hidden=64)  =>
  mx.Activation(name=:relu2, act_type=:relu)   =>
  mx.FullyConnected(name=:fc3, num_hidden=10)  =>
  mx.SoftmaxOutput(name=:softmax)

# data provider
batch_size = 100
include(Pkg.dir("MXNet", "examples", "mnist", "mnist-data.jl"))
train_provider, eval_provider = get_mnist_providers(batch_size)

# setup model
model = mx.FeedForward(mlp, context=mx.gpu())

# optimization algorithm
optimizer = mx.SGD(lr=0.1, momentum=0.9)

# fit parameters
mx.fit(model, optimizer, train_provider, n_epoch=20, eval_data=eval_provider)

You can also predict using the model in the following way:

probs = mx.predict(model, eval_provider)

# collect all labels from eval data
labels = Array[]
for batch in eval_provider
    push!(labels, copy(mx.get(eval_provider, batch, :softmax_label)))
end
labels = cat(1, labels...)

# Now we use compute the accuracy
correct = 0
for i = 1:length(labels)
    # labels are 0...9
    if indmax(probs[:,i]) == labels[i]+1
        correct += 1
    end
end
accuracy = 100correct/length(labels)
println(mx.format("Accuracy on eval set: {1:.2f}%", accuracy))
