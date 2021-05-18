  

  

# Introduction

<div class=text-justify>

What makes a good movie ? It can be the acting, the beauty of the images, the quality of the special effects. But sometimes it is more subtle. The quality of a film is often based on the complexity of the script, the relationships between the characters, the intertwining of their interactions, and the realism that emerges. Do we get attached to the characters, are their interactions well constructed? A film has to create a small coherent universe in which the main and secondary characters will evolve.

With the graph and social network analysis, we can represent the social structure of a movie. The nodes will be the characters and an edge between two characters exists if the characters had an interaction in the movie. The edges are non-weighted. We have an easy access to the ratings of the movie through the IMDb databases. The IMDb ratings are the average of the users’ ratings on the website.



The goal of this work will be to figure out if there is a correlation between the social structure of a movie and its rating on IMDb.  
Then, we will try to see if we can predict the rating of a movie based on its social structure.
</div>

\pagebreak

# Is the social structure of the movie correlated with the rating

First, we will see if the social structure of movies are correlated with their ratings.

## Methodology used 

<div class=text-justify>

To make this work, we have a Database of 808 movies. For each movie, we have a ‘.gexf’ file which is the graph file, and it’s rating on IMDb. To visualise and analyse the social structure of a movie, we can use Gephi. We can have here an individual analysis of each movie. For instance, you can see here the social structure graph of the movie Gladiator.
</div>

![Graph of Gladiator](gladiator.png 'Gladiator')

<div class=text-justify>

I used the 'Force Atlas' algorithm to force the position of the nodes in the graph. And the sizes of the characters are proportional to their importance. There are a lot of metrics that can be used to analyse a node in a graph (like the Degree, the Centrality, ...).  
But for our work, I choose to analyse the graph globally (we'll see later which are the metrics I chose to analyse the movies). I used `NetworkX`, a python framework to analyse all the movies we have in the database.
First, I had to clean a bit the data. I decided to keep only the graphs with more than one node (and at least one edge). And also, some movies didn't have rating in the IMDb database, so I didn't use them.
At the end of the preprocessing of the data, I had 752 movies I can use for my work.  
We have to choose the metrics to represent our movies. We will make vectors of size 752 for each metrics, and we have a vector of size 752 for the ratings. After that, we will calcultate the correlation between the rating and each metric used (with the Speardman correlation coefficient).  
Now, let's chose the metrics to represent the graph of our movies.
</div>


### Metrics to analyse a movie through its graph

#### *Number of nodes and number of edges*  
<div class=text-justify>

When we want to analyse the social structure of a movie, the two simple metrics are the number of characters, and the number of relations between characters. So we have to count the number of nodes and edge of the graph for each movie, and add them into two vectors. 
</div>


```python
nodes = []
edges = []
for graph in graphs: #graphs is the list of the 752 graphs of the movies
    nodes.append(graph.number_of_nodes())
    edges.append(graph.number_of_edges())
```

#### *Densities of the networks*  

It is interesting to measure the density of the network. \\(d = \frac{2m}{n(n-1)}\\), \\(d\\) is the density, \\(n\\) the number of nodes and \\(m\\) the number of edges.
<div class=text-justify>

The density can be interpreted as the following: is the density is close to 1, it means that almost everydoby knows everybody. But if it is close to 0, it means that there are more lonely people, or people that are a bit outside of the story.  
So we calculate the densities and add it to a vector. 
</div>


```python
densities = []
for graph in graphs:
    density = nx.density(graph)
    densities.append(density)
```

#### *Diameters of the networks*  
<div class=text-justify>

It is also interesting to measure the diameter of the network. It is the shortest distance between the two most distant nodes in the network. The interpretation is quite the same as the density but it's still interesting to have this metric.
</div>


```python
diameters = []
for graph in graphs:
    diameter = nx.diameter(graph)
    diameters.append(diameter)
```

#### *Average degree of the network*  
<div class=text-justify>

For one node in the network, the degree is the number of connections it has with other nodes. So, calculating the average degree of a network is a good metric to know if the characters are connected to a lot of other characters or not. It can indicate if the story is complex or not.
</div>


```python
avg_degrees = []
for graph in graphs:
    degrees = dict(graph.degree())
    avg_degree = sum(degrees.values())/graph.number_of_nodes()
    avg_degrees.append(avg_degree)
```

#### *Highest modularity of the network*  
<div class=text-justify>

The modularity measures the strength of division of a network into clusters. So, for a chosen partition of a network, if the modularity is high it means that the nodes from a same cluster are strongly connected, but not with nodes outside of their clusters. So, it can be a good metric to analyse in a complex way the scenario of the movie.
</div>


```python
modularities = []
for graph in graphs:
    graph1 = graph.to_undirected()
    partition = community.best_partition(graph1)
    modularity = community.modularity(partition, graph1)
    modularities.append(modularity)
```

#### *The triadic closure and the highest PageRank of the network*  
<div class=text-justify>

These two metrics could also be interesting to measure in order to see if there are correlated with the ratings.  
The highest PageRank of the network will indicate wether the main character is very central in the story, if he interacts with the other characters or if he is more lonely.
</div>


```python
triadics = []
for graph in graphs:
    triadic = nx.transitivity(graph)
    triadics.append(triadic)

max_pagerank = []
for graph in graphs:
    pr = nx.pagerank(graph)
    max_pr = max(pr.values())
    max_pagerank.append(max_pr)
```

  
  
<div class=text-justify>

So, now we have vectors for all the metrics that could interest us. We will now calculate the correlation between each metric and the ratings.
</div>



### How to measure correlation
<div class=text-justify>

Correlation analysis is used to describe the strength and direction of the relationship between two variables. 
Here, I used the Spearman correlation coefficient (rho) to analyse the correlation of the networks' metrics (number of nodes, number of edges, density, diameter, average degree, etc.) with the rating.

> The Spearman correlation coefficient is a nonparametric measure of rank correlation (statistical dependence between the rankings of two variables). It assesses how well the relationship between two variables can be described using a monotonic function. The Spearman correlation between two variables is equal to the Pearson correlation between the rank values of those two variables; while Pearson's correlation assesses linear relationships, Spearman's correlation assesses monotonic relationships (whether linear or not).
>
> -- <cite>From Wikipedia</cite>

We use this coefficient because we want to see if there are monotonic relationships more than only linear one. 
To know if two variables are correlated, here is a scale we can use:  
* Weak correlation: rho=0.10 to 0.29;  
* Medium correlation: rho=0.30 to 0.49;  
* Strong correlation: rho=0.50 to 1.0  

Now, let's see what are the results of this work
</div>


## Application of these methods to look at the correlation between social structure an rating of a movie

We can calculate the correlation coefficient I presented you just before:


```python
list_of_metrics = [('nodes', nodes), ('edges', edges), ('average degree', avg_degrees), ('diameter', diameters), ('density', densities), ('modularity', modularities), ('triadic closure', triadics), ('max PageRank', max_pagerank)]

from numpy.random import rand
from numpy.random import seed
from scipy.stats import spearmanr

results = {}

for t in list_of_metrics:
    coef, p = spearmanr(ratings, t[1])
    results[t[0]] = coef

results_pd = pd.DataFrame(results, index=[0])
results_pd.head()
```

![Correlations](correlations.png)

<div class=text-justify>

We see that there seems to be no correlation between the metrics we chose and the ratings of movies. Indeed, no correlation coefficient is above 0.1. The most correlated seems to be the density and the modularity.

Let's plot the metrics in function of the ratings to see if we can see some trends.  
*(I'm sorry for the size, you can zoom in to see better)*
</div>

![Metrics = f(rating)](graph1.png)

<div class=text-justify>

It's difficult to see if there are patterns with this representation.  

So, I decided to divide the movies in 10 classes depending on their ratings. The classes are the folowwing: $[0-1[,  [1-2[,  ... ,  [9-10]$. For instance, f a movie has a rating of 7.3, it will be in the class $[7-8[$.  
Then for each class, I calculated the average of the metric considered for the movies in this class.

Here are the results: *(I'm sorry for the size, you can zoom in to see better)*
</div>

![Metrics = f(rating)](graph2.png)

<div class=text-justify>

These figures are more understandable. and we can see that there are some patterns.  
In general, movies have better ratings if the number of characters and links between characters are higher. (This is just not true here for movies with ratings between 2 and 3).   
We can see also that in general, the rating increase with the diameter of the network. (Here, it is just not true for movies between 3 and 4).  
It is quite the same for modularity.  
Also, what is interesting to see is that the best movies don't have such a high max PageRank. This mean that there a not a very central character that we see all the time on screen, having links with all the other characters.  

So, we can see that, in general, people prefers movies with a lot of characters and links between them, with complex stories, but not necessarily a very central character. (Of course, this conclusion is very general, and it is not true for all movies.)  
  
  
We saw that we can detect some correlations between the social structure of a movie and its rating. But these metrics won't be enough to predict the rating of a movie from its social structure. We could try to predict this with a method I present you next. 
</div>

\pagebreak

# Can we predict the rating of the movie with automated learning?

<div class=text-justify>

Here we will try to predict the rating of the movie based on a vector representation of a graph. It will be a classification problem:  
The input will be that vector reprensentating the movie. We could use a vector based on the metrics we saw before. But we will use the graph2vec method to build a vector from a graph.  
The output of this classification problem will be 0 or 1: 0 means that the rating of the movie is under the average rating, while 1 means that the rating is above the average.  
Now, let's see how to represent our graph with a vector.
</div>




## An other way to represent a graph (graph2vec)


I used the graph2vec method as it is described here:

>graph2vec: Learning distributed representations of graphs. Narayanan, Annamalai and Chandramohan, Mahinthan and Venkatesan, Rajasekar and Chen, Lihui and Liu, Yang MLG 2017, 13th International Workshop on Mining and Learning with Graphs (MLGWorkshop 2017).
>
> *https://www.researchgate.net/publication/318489171_graph2vec_Learning_Distributed_Representations_of_Graphs*

<div class=text-justify>
In this article, we learn that graph2vec is the first neural embedding approach that learns representations of whole graphs offering the following advantages: Unsupervised representation learning, Task-agnostic embeddings, Data-driven embeddings, Captures structural equivalence.

To use graph2vec in my work, I used a python framework called `KarateClub`. It is an unsupervised machine learning extension library for NetworkX.  
I started by building my model and at the end we get for each movie graph a vector of size 128.
</div>




```python
# building new graphs adapted to graph2vec: ordered indexes starting from 0 for the nodes, and undirected graphs
graphs1 = []
for graph in graphs:
    node_dict = {}
    i = 0
    nodes = list(graph.nodes)
    for node in nodes:
        node_dict[node] = i
        i+=1
    graph1 = nx.relabel_nodes(graph, node_dict, copy=True)
    graphs1.append(graph1.to_undirected())

from karateclub.graph_embedding import Graph2Vec

model = Graph2Vec()
model.fit(graphs1)
graphs_vec = model.get_embedding()

```

## Is the rating of the movie predictable ?

<div class=text-justify>

Now we have this, we will train a SVM to try to classify the movies.  
First, we build a `hot_rating` vector only with 0 or 1 depending on if the rating is under or above the average rating.
</div>


```python
import statistics

avg_rating = statistics.mean(ratings)
hot_ratings = []
for x in ratings:
    if x > avg_rating:
        hot_ratings.append(1)
    else:
        hot_ratings.append(0)
```

<div class=text-justify>

Then, we build our SVM with a sigmoid kernel.
</div>


```python
from sklearn.model_selection import train_test_split
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(graphs_vec, hot_ratings, test_size=0.2, random_state=42)

my_svm = svm.SVC(kernel='sigmoid')
my_svm.fit(X_train, y_train)
```

After that, we can try to predict the ratings of the test dataset and measure the accuracy.


```python
from sklearn import metrics

y_pred = my_svm.predict(X_test)
print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred))
```

The output here is `Accuracy: 0.5298013245033113`.  
So, the SVM's accuracy is just above 50%. It is not very good.

\pagebreak

# Discussion

<div class=text-justify>

The results we got here are not very impressive.  

For the first part may be I could have try to chose better metrics. For instance, I could have look to the connected components: is there only a giant connected component or are there multiples components in the graph?  
I could also look at the node's metrics to be more precise.
An other improvement possibility could have be on the graph. Maybe if the edges would have been weighted, we could have improved our results.

For the second part, when I used graph2vec, I could have other parameters, or other methods for graph embeddings, but I didn't have enough time. 
I also tried to make the classification with a logistic regression but it was not better.  
Maybe with a bigger dataset, I could have tried to build a Neural Network to predict the rating. But here we don't have enough data to do so.

</div>

 
\pagebreak

# Conclusion

<div class=text-justify>

During this work, we saw that we can detect correlation between some metrics of the movies' graphs and their ratings. In general, we saw that people prefer a movie with more complex story, with a lot of characters, and a lot of links between them. Also, movies don't have to have a very central character to be a good movie.  
Is it possible to find a correlation between social structures and ratings of movies, but we have to chose wisely how we want to represent the social networks of the movies. Here I made choices about the metrics to describe them, but there can be other choice with which we will see a higher correlation.

We also tried build a classifier to predict the rating according to the social structure of a movie. We could have build a neural network, which could have been more efficient, bit we didn't have enough data to train it, and we would have faced an overfitting problem.

Even if the results are not as good as we could have hope, it is very interesting that there are patterns in the movies that people love, and I'm sure that the rating could be predicted quite well with a bigger dataset to train a neural network.
</div>


*Link to the Github repository with the code I used for my work: https://github.com/19defroide/Graph_SNA_Practical_Work*
