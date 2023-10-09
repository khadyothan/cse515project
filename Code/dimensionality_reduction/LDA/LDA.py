from pymongo import MongoClient
import sys
import numpy as np
from pprint import pprint
import gensim
import json
#Connecting to mongoDB client and creating database and collections
client = MongoClient("localhost", 27017)
DB = client.caltech101db
collection = DB.caltech101withGSimages

#extracting features from the database and storing in lists
def retrieve_Features(collection, feature_space):
    feature_list = []
    feature_list=[i[feature_space] for i in collection.find({}, {feature_space: 1}) ]
    #normalizing HOG feature values
    if(feature_space in ('hog_feature_descriptor')):
        for i in range(len(feature_list)):
            f1=feature_list[i]
            f1=np.array(f1).reshape((1,900)).flatten()
            min_val=f1.min()
            if(min_val<0):
                f1=f1-min_val
            max_val=f1.max()
            f1=f1/max_val
            feature_list[i]=f1
    
    #normalizing Colormoments feature values
    elif(feature_space in ('color_moments_feature_descriptor')):
        #extracting minimum value for each mean, SD, Skewness
        for i in range(len(feature_list)):
            f1=feature_list[i]
            min_vals=[sys.maxsize]*3
            for j in f1:#iterating through images
                for k in j:#iterating through grids
                    for l in k:#iterating through channels
                        for idx,ele in enumerate(l):#iterating through 3 color_moments(mean,SD,Skewness)
                            if(min_vals[idx]>ele):
                                min_vals[idx]=ele

            #identifying which color_moment (mean/SD/Skewness) have negative minimum values
            idxes_to_be_added=[]
            for j in range(3):
                if(min_vals[j]<0):
                    idxes_to_be_added.append(j)

            #adding -(negative min value of each color_moment found) to all values belonging to that color moment to help normalize the values 
            for j in f1:#iterating through images
                for k in j:#iterating through grids
                    for l in k:#iterating through channels
                        for idx in idxes_to_be_added:#iterating through 3 color_moments(mean,SD,Skewness)
                            l[idx]+=-(min_vals[idx])

            min_vals=[sys.maxsize]*3
            max_vals=[-sys.maxsize+1]*3
            #identifying the maximum value of each color_moment
            for j in f1:#iterating through images
                for k in j:#iterating through grids
                    for l in k:#iterating through channels
                        for idx,ele in enumerate(l):#iterating through 3 color_moments(mean,SD,Skewness)
                            if(max_vals[idx]<ele):
                                max_vals[idx]=ele
                                
            #divinding by maximum value to normalize each color_moment
            for j in f1:#iterating through images
                for k in j:#iterating through grids
                    for l in k:#iterating through channels
                        for idx in range(len(l)):#iterating through 3 color_moments(mean,SD,Skewness)
                            l[idx]/=max_vals[idx]
            f1=np.array(f1).reshape((1,900)).flatten()
            feature_list[i]=f1

    return feature_list

#function that takes starting id and feature space as parameters and produces a id mapping for all feature values
def create_id2num(current_id, feature_space):
    id2num = dict()
    for feature in feature_space:
        for i in feature:
            rounded_val = round(i,2)
            if rounded_val in id2num.values():
                continue
            else:
                id2num[current_id] = rounded_val
                current_id+=1
    return id2num

#returns the id of a feature value from the passed id2num mapping
def getkey(i,id2num):
    for j,k in id2num.items():
        if k == i:
            return j

#Creates a corpus of all the features in the feature_space
def create_corpus(id2num, feature_space):
    corpus = []
    for feature in feature_space:
        temp_dict = dict()
        for i in feature:
            id = getkey(round(i,2),id2num)#rounding off each value of the selected feature descriptor to make them discrete
            if id not in temp_dict.keys():
                temp_dict[id] = 1
            else:
                temp_dict[id]+=1
        corpus.append(list(temp_dict.items()))
    return corpus


def retrieve_LDA_latent_samantics(feature_descriptor,k):
    features  = retrieve_Features(collection,feature_descriptor)
    start_index = {"color_moments_feature_descriptor":1,"hog_feature_descriptor":2000,
                   "resnet50_layer3_feature_descriptor":4000,"resnet50_avgpool_feature_descriptor":6000,
                   "resnet50_fc_feature_descriptor":8000}
    id2num = create_id2num(start_index[feature_descriptor],features)
    corpus = create_corpus(id2num,features)
    # Build LDA model
    print("im here")
    lda_model = gensim.models.LdaMulticore(corpus=corpus,id2word=id2num,num_topics=k)
    print("LDA completed")
    # Print the Keyword in the 10 topics
    #pprint(lda_model.print_topics())
    topic_distribution = lda_model.print_topics(num_words=30)
    latent_semantics = dict()
    for i in range(k):
        latent_semantics[topic_distribution[i][0]] = topic_distribution[i][1]
    file_name = feature_descriptor + "latentSemantics.json"
    with open(file_name, "w") as outfile:
        json.dump(latent_semantics, outfile)

    doc_topics = lda_model.get_document_topics(corpus)
    image_weight_matrix = []
    for img in doc_topics:
        temp_dict = dict()
        for i in range(k):
            temp_dict[i] = 0
        for topic,weight in img:
            temp_dict[topic] = weight
        image_weight_matrix.append(sorted(temp_dict.items(), key=lambda x:x[1],reverse=True))
    print(image_weight_matrix)
        
retrieve_LDA_latent_samantics("resnet50_layer3_feature_descriptor",10)