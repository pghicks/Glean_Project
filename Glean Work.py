#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance
import string
from re import search
 
# Create initial embeddings dictionary
def get_embeddings_dict(embeddings_dict):
    # Open and create dict
    with open(directory + "glove.6B.50d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            # Split line by space
            values = line.split()
            # Get word
            word = values[0]
            # Make array for embeddings
            vector = np.asarray(values[1:], "float32")
            # Update dict
            embeddings_dict[word] = vector
    return(embeddings_dict)

# Create idfs
def get_df_idf():
    word_list = list(line_items["canonical_line_item_name"])
    no_integers = [x for x in word_list if not isinstance(x, int)]

    # Instantiate CountVectorizer()
    cv=CountVectorizer()

    # Generate word counts
    word_count_vector=cv.fit_transform(no_integers)

    # Transform to idf values
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    # Create idf values df
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
    return(df_idf)

# Finds embeddings for each cannonical
def get_canonical_embeddings(line_items):
    em_dim = 50
    # Create empty matrix
    embedding = np.mat([0] * em_dim * len(line_items))
    embedding = embedding.reshape(len(line_items), em_dim)
    line_items = pd.concat([line_items, pd.DataFrame(embedding)], axis=1)

    for i in range(len(line_items)):
        p = 0
        # Removes hourly services as many canonicals add them in addition to line item names
        line = line_items["canonical_line_item_name"][i].translate(str.maketrans('', '', string.punctuation)).replace("Hourly Services:", "").split()
        for word in line:
            # Don't include words that don't have embeddings
            try:
                # Get idf score for word
                idf = df_idf.loc[word.lower()]
                length = len(line_items["canonical_line_item_name"][i].split())
                # Creates weight which favors earlier words
                weight = (length - p)/length
                # Adds word embedding times weight and id to existing sentence embedding
                line_items.loc[[i], line_items.columns[2:]] = line_items.loc[[i], line_items.columns[2:]].add(embeddings_dict[word.lower()]*float(idf)*weight)
                p += 1
            except:
                pass
    return(line_items)

# Makes embedding for line
def make_embedding(line, q):
    p = 0
    embedding = 0
    for word in line:
        try:
            idf = df_idf.loc[word.lower()]
            length = len(line)
            weight = (length - p)/length
            embedding += embeddings_dict[word.lower()]*float(idf)*weight
            p += 1
        except:
                pass
    return(embedding)

# Finds distance between embeddings
def get_distance(embedding, q, same_vendor):
    min_distance = 100000000
    min_distance2 = 100000000
    cat = ""
    i = 0
    for row in same_vendor:
        distance = np.linalg.norm(embedding-row[1][2:])
        if distance < min_distance:
            min_distance2 = min_distance
            min_distance = distance
            cat = i
        i += 1
    return(cat, min_distance, min_distance2)

# Finds accuracy on training data
def get_train_acc():
    correct = [0] * len(train)
    for q in range(len(train)):
        match = 0
        # Find all canonical options for vendor
        vendor_rows = line_items["canonical_line_item_name"].loc[line_items["canonical_vendor_name"] == train["canonical_vendor_name"][q]]
        # Select if there's only one option
        if len(vendor_rows) == 1:
            estimate = vendor_rows.iloc[0]
            match = 1
            # See if estimate is correct
            if estimate == train.loc[q, "canonical_line_item_name"]:
                correct[q] = 1
        if match == 1:
            continue   
            
        # See if canonical is identical to line item name
        for row in line_items["canonical_line_item_name"].loc[line_items["canonical_vendor_name"] == train["canonical_vendor_name"][q]]:
            if row == train.loc[q, "line_item_name"]:
                estimate = row
                match = 1
                if estimate == train.loc[q, "canonical_line_item_name"]:
                    correct[q] = 1

        if match == 1:
            continue  
        
        # Some cases didn't work well with embeddings, a few lines of logic were used to help
        # Filter cases for Amelia Willson
        if train["canonical_vendor_name"][q] == "Amelia Willson":
            if search("900-1,500 words", train["line_item_description"].loc[q]) or search("900-1,500 words", train["line_item_name"].loc[q]):
                match = 1
                estimate = "900-1,500 words"
            elif search("1,500-2,000 words", train["line_item_description"].loc[q]) or search("1,500-2,000 words", train["line_item_name"].loc[q]):
                match = 1
                estimate = "1,500-2,000 words"
            else:
                match = 1
                estimate = "Trial Assignment"
            if estimate == train.loc[q, "canonical_line_item_name"]:
                correct[q] = 1
        
        # Filter cases for Graphite Financial
        if train["canonical_vendor_name"][q] == "Graphite Financial":
            if search("Discounts / Credits Acct", train["line_item_name"].loc[q]):
                match = 1
                estimate = "Discounts / Credits Acct (e)"
            # Make sure line_item_description exists
            elif str(train["line_item_description"].loc[q]) != "nan":
                if search("PROJECT", str(train["line_item_description"].loc[q])):
                    match = 1
                    estimate = "Hourly Services: Projects"
                elif search("Accounting:Core_Accounting_Service", train["line_item_name"].loc[q]):
                    match = 1
                    estimate = "Hourly Services: Tasks"
            if search("Strategic Finance Team", train["line_item_name"].loc[q]):
                match = 1
                estimate = "Hourly Services: Strategic Finance Team"  
            if estimate == train.loc[q, "canonical_line_item_name"]:
                correct[q] = 1
                
        # Filter cases for Maddie Shepherd
        # If name isn't identical to canonical, then its a blog post
        if train["canonical_vendor_name"][q] == "Maddie Shepherd":
            match = 1
            estimate = "Blog Post"  
            if estimate == train.loc[q, "canonical_line_item_name"]:
                correct[q] = 1

        # Filter cases for Andersen Tax
        if train["canonical_vendor_name"][q] == "Andersen Tax":
            if search("Director", train["line_item_description"].loc[q]):
                match = 1
                estimate = "Hourly Services: Legal Services"
            else:
                match = 1
                estimate = "Hourly Services: Tax Services"
            if estimate == train.loc[q, "canonical_line_item_name"]:
                correct[q] = 1
                
        # Filter cases for Daversa Partners
        if train["canonical_vendor_name"][q] == "Daversa Partners":
            match = 1
            estimate = "Retainer: " + train["line_item_name"].loc[q] 
            if estimate == train.loc[q, "canonical_line_item_name"]:
                correct[q] = 1

        # Filter cases for Westmont Associates
        if train["canonical_vendor_name"][q] == "Westmont Associates":
            if search("Buzzy", train["line_item_description"].loc[q]):
                match = 1
                estimate = "Expenses: Buzzy P&C and Surplus Line Renewal"
            # All others are Expenses: Filing Fees
            elif train["line_item_name"].loc[q] == "Expenses":
                match = 1
                estimate = "Expenses: Filing Fees"
            elif train["line_item_name"].loc[q] == "Flat Fee":
                match = 1
                estimate = "Non-Hourly Services: A&A"
            if estimate == train.loc[q, "canonical_line_item_name"]:
                correct[q] = 1
                
        # Filter cases for Xiamen ZhiZhi Tech
        if train["canonical_vendor_name"][q] == "Xiamen ZhiZhi Tech":
            if search("Misc", train["line_item_name"].loc[q]):
                match = 1
                estimate = "Misc. expenses"
            if estimate == train.loc[q, "canonical_line_item_name"]:
                correct[q] = 1
        if match == 1:
            continue

        # Remove punctuation and split by word
        line = str(train.loc[q, "line_item_name"]).translate(str.maketrans('', '', string.punctuation)).split()
        # Create embedding for line
        embedding = make_embedding(line, q)
        # Find canonical options for vendor
        vendor_rows = line_items.loc[line_items["canonical_vendor_name"] == train["canonical_vendor_name"][q]].iterrows() 
        # Find distance between line embedding and canonical possibilities
        cat, min_distance, min_distance2 = get_distance(embedding, q, vendor_rows)

        # Make sure its accurate and clear winner
        # Next phases use line_item_description, need to make sure it exists
        if (min_distance < 25 and min_distance2 - min_distance > 5) or str(train.loc[0, "line_item_description"]) == 'nan':
            # Estimate is closest option
            estimate = line_items["canonical_line_item_name"].loc[line_items["canonical_vendor_name"] == train["canonical_vendor_name"][q]].iloc[cat]
            # See if correct
            if estimate == train.loc[q, "canonical_line_item_name"]:
                correct[q] = 1

        # Run second phase looking at line_item_name/line_item_description combo
        else:
            embedding = 0
            p = 0
            # Concat phrases
            line = str(train.loc[q, "line_item_name"]) + " " + str(train.loc[q, "line_item_description"])
            # Remove punctuation
            line = line.translate(str.maketrans('', '', string.punctuation)).split()
            # Create embedding for line
            embedding = make_embedding(line, q)
            # Find canonical options for vendor
            vendor_rows = line_items.loc[line_items["canonical_vendor_name"] == train["canonical_vendor_name"][q]].iterrows()
            # Find distance between line description embedding and canonical possibilities
            cat, min_distance, min_distance2 = get_distance(embedding, q, vendor_rows)

            # Make sure its accurate and clear winner
            if (min_distance < 40 and min_distance2 - min_distance > 5)or str(train.loc[0, "line_item_description"]) == 'nan':
                # Estimate is closest option
                estimate = line_items["canonical_line_item_name"].loc[line_items["canonical_vendor_name"] == train["canonical_vendor_name"][q]].iloc[cat]
                # See if correct
                if estimate == train.loc[q, "canonical_line_item_name"]:
                    correct[q] = 1
            # Run third phase only using line_item_description
            else:
                embedding = 0
                p = 0
                # Remove punctuation
                line = str(train.loc[q, "line_item_description"])
                # Remove punctuation
                line = line.translate(str.maketrans('', '', string.punctuation)).split()
                # Create embedding for line
                embedding = make_embedding(line, q)
                # Find canonical options for vendor
                vendor_rows = line_items.loc[line_items["canonical_vendor_name"] == train["canonical_vendor_name"][q]].iterrows()
                # Find distance between line description embedding and canonical possibilities
                cat, min_distance, min_distance2 = get_distance(embedding, q, vendor_rows)
                
                # Make sure its accurate and clear winner
                if (min_distance < 20 and min_distance2 - min_distance > 2)or str(train.loc[0, "line_item_description"]) == 'nan':
                    # Estimate is closest option
                    estimate = line_items["canonical_line_item_name"].loc[line_items["canonical_vendor_name"] == train["canonical_vendor_name"][q]].iloc[cat]
                    # See if correct
                    if estimate == train.loc[q, "canonical_line_item_name"]:
                        correct[q] = 1
                        
                # If all three phases produce no clear estimate, none will be selected
                else:
                    correct[q] = None

    print("Correct: " + str(correct.count(1)))
    print("Incorrect: " + str(correct.count(0)))
    print("Unsure: " + str(correct.count(None)))
    return(correct)

# Fills in evaldf
def fill_evaldf(evaldf):
    correct = [0] * len(evaldf)
    for q in range(len(evaldf)):
        match = 0
        # Find all canonical options for vendor
        vendor_rows = line_items["canonical_line_item_name"].loc[line_items["canonical_vendor_name"] == evaldf["canonical_vendor_name"][q]]
        # Select if there's only one option
        if len(vendor_rows) == 1:
            match = 1
            evaldf["canonical_line_item_name"].iloc[q] = vendor_rows.iloc[0]
        if match == 1:
            continue   
            
        # See if canonical is identical to line item name
        for row in line_items["canonical_line_item_name"].loc[line_items["canonical_vendor_name"] == evaldf["canonical_vendor_name"][q]]:
            if row == evaldf.loc[q, "line_item_name"]:
                evaldf["canonical_line_item_name"].iloc[q] = row
                match = 1

        if match == 1:
            continue  
        
        # Some cases didn't work well with embeddings, a few lines of logic were used to help
        # Filter cases for Amelia Willson
        if evaldf["canonical_vendor_name"][q] == "Amelia Willson":
            if search("900-1,500 words", evaldf["line_item_description"].loc[q]) or search("900-1,500 words", evaldf["line_item_name"].loc[q]):
                match = 1
                evaldf["canonical_line_item_name"].iloc[q] = "900-1,500 words"
            elif search("1,500-2,000 words", evaldf["line_item_description"].loc[q]) or search("1,500-2,000 words", evaldf["line_item_name"].loc[q]):
                match = 1
                evaldf["canonical_line_item_name"].iloc[q] = "1,500-2,000 words"
            else:
                match = 1
                evaldf["canonical_line_item_name"].iloc[q] = "Trial Assignment"
        
        # Filter cases for Graphite Financial
        if evaldf["canonical_vendor_name"][q] == "Graphite Financial":
            if search("Discounts / Credits Acct", evaldf["line_item_name"].loc[q]):
                match = 1
                evaldf["canonical_line_item_name"].iloc[q] = "Discounts / Credits Acct (e)"
            # Make sure line_item_description exists
            elif str(evaldf["line_item_description"].loc[q]) != "nan":
                if search("PROJECT", str(evaldf["line_item_description"].loc[q])):
                    match = 1
                    evaldf["canonical_line_item_name"].iloc[q] = "Hourly Services: Projects"
                elif search("Accounting:Core_Accounting_Service", evaldf["line_item_name"].loc[q]):
                    match = 1
                    evaldf["canonical_line_item_name"].iloc[q] = "Hourly Services: Tasks"
            if search("Strategic Finance Team", evaldf["line_item_name"].loc[q]):
                match = 1
                evaldf["canonical_line_item_name"].iloc[q] = "Hourly Services: Strategic Finance Team" 
                
        # Filter cases for Maddie Shepherd
        # If name isn't identical to canonical, then its a blog post
        if evaldf["canonical_vendor_name"][q] == "Maddie Shepherd":
            match = 1
            evaldf["canonical_line_item_name"].iloc[q] = "Blog Post"  

        # Filter cases for Andersen Tax
        if evaldf["canonical_vendor_name"][q] == "Andersen Tax":
            if search("Director", evaldf["line_item_description"].loc[q]):
                match = 1
                evaldf["canonical_line_item_name"].iloc[q] = "Hourly Services: Legal Services"
            else:
                match = 1
                evaldf["canonical_line_item_name"].iloc[q] = "Hourly Services: Tax Services"
                
        # Filter cases for Daversa Partners
        if evaldf["canonical_vendor_name"][q] == "Daversa Partners":
            match = 1
            evaldf["canonical_line_item_name"].iloc[q] = "Retainer: " + evaldf["line_item_name"].loc[q] 

        # Filter cases for Westmont Associates
        if evaldf["canonical_vendor_name"][q] == "Westmont Associates":
            # Make sure line_item_description exists
            if str(evaldf["line_item_description"].loc[q]) != "nan":
                if search("Buzzy", evaldf["line_item_description"].loc[q]):
                    match = 1
                    evaldf["canonical_line_item_name"].iloc[q] = "Expenses: Buzzy P&C and Surplus Line Renewal"
                # All others are Expenses: Filing Fees    
                elif evaldf["line_item_name"].loc[q] == "Expenses":
                    match = 1
                    evaldf["canonical_line_item_name"].iloc[q] = "Expenses: Filing Fees"
                elif evaldf["line_item_name"].loc[q] == "Flat Fee":
                    match = 1
                    evaldf["canonical_line_item_name"].iloc[q] = "Non-Hourly Services: A&A"

            # All others are Expenses: Filing Fees
            elif evaldf["line_item_name"].loc[q] == "Expenses":
                match = 1
                evaldf["canonical_line_item_name"].iloc[q] = "Expenses: Filing Fees"
            elif evaldf["line_item_name"].loc[q] == "Flat Fee":
                match = 1
                evaldf["canonical_line_item_name"].iloc[q] = "Non-Hourly Services: A&A"

                
        # Filter cases for Xiamen ZhiZhi Tech
        if evaldf["canonical_vendor_name"][q] == "Xiamen ZhiZhi Tech":
            if search("Misc", evaldf["line_item_name"].loc[q]):
                match = 1
                evaldf["canonical_line_item_name"].iloc[q] = "Misc. expenses"
        if match == 1:
            continue

        # Remove punctuation and split by word
        line = str(evaldf.loc[q, "line_item_name"]).translate(str.maketrans('', '', string.punctuation)).split()
        # Create embedding for line
        embedding = make_embedding(line, q)
        # Find canonical options for vendor
        vendor_rows = line_items.loc[line_items["canonical_vendor_name"] == evaldf["canonical_vendor_name"][q]].iterrows() 
        # Find distance between line embedding and canonical possibilities
        cat, min_distance, min_distance2 = get_distance(embedding, q, vendor_rows)

        # Make sure its accurate and clear winner
        # Next phases use line_item_description, need to make sure it exists
        if (min_distance < 25 and min_distance2 - min_distance > 5) or str(evaldf.loc[0, "line_item_description"]) == 'nan':
            # Estimate is closest option
            evaldf["canonical_line_item_name"].iloc[q] = line_items["canonical_line_item_name"].loc[line_items["canonical_vendor_name"] == evaldf["canonical_vendor_name"][q]].iloc[cat]

        # Run second phase looking at line_item_name/line_item_description combo
        else:
            embedding = 0
            p = 0
            # Concat phrases
            line = str(evaldf.loc[q, "line_item_name"]) + " " + str(evaldf.loc[q, "line_item_description"])
            # Remove punctuation
            line = line.translate(str.maketrans('', '', string.punctuation)).split()
            # Create embedding for line
            embedding = make_embedding(line, q)
            # Find canonical options for vendor
            vendor_rows = line_items.loc[line_items["canonical_vendor_name"] == evaldf["canonical_vendor_name"][q]].iterrows()
            # Find distance between line description embedding and canonical possibilities
            cat, min_distance, min_distance2 = get_distance(embedding, q, vendor_rows)

            # Make sure its accurate and clear winner
            if (min_distance < 40 and min_distance2 - min_distance > 5) or str(evaldf.loc[0, "line_item_description"]) == 'nan':
                # Estimate is closest option
                evaldf["canonical_line_item_name"].iloc[q] = line_items["canonical_line_item_name"].loc[line_items["canonical_vendor_name"] == evaldf["canonical_vendor_name"][q]].iloc[cat]
    
            # Run third phase only using line_item_description
            else:
                embedding = 0
                p = 0
                # Remove punctuation
                line = str(evaldf.loc[q, "line_item_description"])
                # Remove punctuation
                line = line.translate(str.maketrans('', '', string.punctuation)).split()
                # Create embedding for line
                embedding = make_embedding(line, q)
                # Find canonical options for vendor
                vendor_rows = line_items.loc[line_items["canonical_vendor_name"] == evaldf["canonical_vendor_name"][q]].iterrows()
                # Find distance between line description embedding and canonical possibilities
                cat, min_distance, min_distance2 = get_distance(embedding, q, vendor_rows)
                
                # Make sure its accurate and clear winner
                if (min_distance < 20 and min_distance2 - min_distance > 2)or str(evaldf.loc[0, "line_item_description"]) == 'nan':
                    # Estimate is closest option
                    evaldf["canonical_line_item_name"].iloc[q] = line_items["canonical_line_item_name"].loc[line_items["canonical_vendor_name"] == evaldf["canonical_vendor_name"][q]].iloc[cat]
                        
                # If all three phases produce no clear estimate, none will be selected
                else:
                    evaldf["canonical_line_item_name"].iloc[q] = None

    return(evaldf)


# In[18]:


# Load Excel Files
# Change to directory 
directory = "/Users/philliphicks/Desktop/Glean Project/"
train = pd.read_excel(directory + "question-python-data-science-project-mwsr7tgbeo-mapping_challenge.xlsx", sheet_name = "train")
evalframe = pd.read_excel(directory + "question-python-data-science-project-mwsr7tgbeo-mapping_challenge.xlsx", sheet_name = "eval")
line_items = pd.read_excel(directory + "question-python-data-science-project-mwsr7tgbeo-mapping_challenge.xlsx", sheet_name = "canonical_line_item_table")

embeddings_dict = {}
# Download the pretrained 6B vectors at https://nlp.stanford.edu/projects/glove/
embeddings_dict = get_embeddings_dict(embeddings_dict)
# Get idf data frame
df_idf = get_df_idf()
# Finds embeddings for each canonical
line_items = get_canonical_embeddings(line_items)
# Finds accuracy for training set
correct = get_train_acc()
# Fills in evaldf
evalframe = fill_evaldf(evalframe)
# Writes to csv
evaldf.to_csv(directory + "answers.csv")

