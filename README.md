# Steam Project: Recommendation System Based on Cosine Similarity

The purpose of this project is to simulate a MLOps task, from ETL to the deployment of and API to a free cloud service.

The entire project structure follows a straighforward path, from reading and cleaning data to Eploratory Analysis, following by
the development of Machine Learning models with the objective of building a Recommendation System. Finally, clean and processed data
is available to query from a Web API created with FastAPI and deployed in Render.

## **Walkthrough**

The development of this project goes through different stages. Detailed procedures can be found in the [Notebooks](./Notebooks/) folder.

Note: The order of the jupyter notebooks show exactly the approach taken, so it is important to follow it carefully in case reviewing them.

### **ETL Stage**

- Data provided is stored in the [raw_data](./raw_data/) folder. There are 3 json files.
- Some of the files contain nested objects and disctinct formats for the data. The **first task** was to read the data accordingly.
- Some transformations were made. General data cleaning (Dropping Nulls, Unpacking nested objects, some columns dropped).

The files where heavy and there were more than 5 million json nested objects contained in ['users_items.json.gz](./raw_data/users_items.json.gz). **Reading and unpacking this files was the main challenge in this stage**

### **EDA Stage**

Once data is cleaned and ready to consume, it can be explored to gain insights and defining what features are the most informative and useful to build the Recommendation System.

- **Sentiment Analysis** was performed to the [reviews](./data/reviews.csv.gz) dataset. It contains more than 50 thousand reviews for different items. The sentiment score was achieved by using the NLTK module `VADER`, which perfoms some kind of 'unsupervised' approach to compute scores (it doesn't need to be trained). The trade-off for the simplicity and easy-usage of `VADER` is that scores could not be too accurate.

- After every single review was labeled as 'Positive', 'Negative' or 'Neutral', data can be grouped and analysed by complex joins (every dataset has a column 'item_id') and queries.

- At this point, the API endpoints were created as it only required having clean, unnested and readable data.

- Criteria for similarity was defined.

- Feature Selection for building the recommender model.

- A **Pre-processing** baseline is determined.

Note: Development details of this stage on Notebooks 2 to 4.

### **Machine Learing Stage: Recommendation System**

Some theory: From all the techniques used by recommender systems, there is one of the most common and classical distinctions, that involves two popular approaches: 

- '**Colaborative Filtering**' is a method to compute interests based on **user-item** interactions. It is about filtering the information by preferences of distinct users. It can be seen as: "Users similar to you also liked: (...)".

- '**Content Based**' approach uses different techniques to compute similarities between items based on content (or attributes). It is based on **item-item** interaction. It can be seen as: "Similar objects to this: (...)"

**In this project the Content-Based approach was choosen.** 

The objective is to make game recomendations based on similarity, using the **Cosine Similarity** algorithm to compute similarity scores between itmes.

- A preprocess function was created. It prepares a copy of the games data, transforming it into a higher dimensional sparse matrix (most of its items are zero).

- As an analogy of the 'Estimator' class from [Sckit-Learn](https://scikit-learn.org/stable/index.html), a 'Similarity Computer' class was created.

    - `CosSinComputer` class contains the recmmender system. The need of using mulitple attributes and methods to make recommendations leads to writing long and complex blocks of code. Creating a class object can solve most problems as i must use distinct methods and procedures to compute similirities and creating responses.

    - It is simply taking the idea of `Estimators` from scikit-learn. For sure, this "`Computer`" object is way simpler and less sofisticated than any class from any Machine Learning library, but at least it perform simple yet useful calculations specifically for the purpose of this project.

    - `CosSinComputer` can be "trained" with a pandas DataFrame; which is an ilusion because it just stores the DataFrame as an instance attribute.

    - It has an equivalent for a "predict" method; `CosSinComputer.compute_similarities()` performs the Cosine Similarity algorithm (from Scikit-learn) to compute similarities between vectors.
