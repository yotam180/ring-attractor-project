Research Proposal – How Partial Observation Affects the Ability of RNN to Reconstruct Ring Attractor Dynamics
Background
RNNs can be used as surrogate DS models for neural population activity. After training an RNN to reproduce recorded activity, we can use analyze the trained network to infer underlying mechanisms (e.g. continuous attractor manifolds vs. discrete stable states).
However, in real experiments, we only have access to a subset of the true data from the neural system, which can sometimes lead to models recovering an incorrect internal mechanism, fitting the data well but not catching the underlying dynamics properly.
Objective
Quantify when an RNN trained on partial spiking network data can correctly recover the dynamics of the underlying system, and when it produces a plausible but incorrect explanation of it.
We then apply the validated pipeline to real neural recordings and report the inferred dynamics with a calibrated measure of “uncertainty”.
Assumptions
H1
If we record enough neurons (\frac{N*{observed}}{N*{total}} high enough) with a long enough time period (T*{end}-T*{start} long enough), the RNN will recover the right dynamical class. That means, the model, under given noise input, will move through various values of \theta in a continuous way, rather than settle into a discrete number of points \theta_1,\theta_2\ldots
The point – if the “bump” in the RNN never moves and only settles into one discrete location, then the RNN has learnt no ring attractor dynamics.
H2
If we record only a subset of neurons, the RNN can fit perfectly but infer the wrong dynamical class. That might appear, for example, as if the RNN has settled into point attractor mechanism, or a “discretized” ring that contains a number of fixed points but no drift/manifold mechanisms.
We predict that below some threshold in the number of neurons or time scale the RNN, its predictive accuracy (ability to reproduce the observed spike dynamics) stays high, although the dynamical class reproduction ability will be decreased (checked by analyzing fixed points, perturbation responses).
Approach
Teacher simulation – simulate a ring attractor network and use it to generate spiking data.
Data loss simulation – in each experiment, model the neuron data dropout and pick different timespans for simulation, leaving us with data of various degrees of “completeness”.
Data preparations – smooth out the data into time buckets of known dt.
Student network training – we train RNNs on the spike rate inputs and evaluate the trained networks in their predictive qualities (“how well data from the RNN simulates original data”)
Student network mechanistic analysis – find fixed points from various initial conditions on each network, perform perturbation tests to observe movement and degrees of freedom, topology, and set a criterion for when it “matches” a ring-like attractor.
Real data application – apply steps 2-5 to a real dataset and review the inferred mechanical classification as a function of the applied data filtering aggressiveness.
