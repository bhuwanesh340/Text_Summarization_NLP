#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import re
import string
import pandas as pd
from functools import reduce
from math import log

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import math

from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords   

from nltk.cluster import KMeansClusterer
from sklearn import cluster


# # Define functions for Word Frequency, TF, IDF and finally TF-IDF matrix creation

# In[2]:


def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            
            word=re.sub("(\\t)", ' ', str(word)).lower() #remove escape charecters
            word=re.sub("(\\r)", ' ', str(word)).lower() 
            word=re.sub("(\\n)", ' ', str(word)).lower()

            word=re.sub("(__+)", '', str(word)).lower()   #remove _ if it occors more than one time consecutively
            word=re.sub("(--+)", '', str(word)).lower()   #remove - if it occors more than one time consecutively
            word=re.sub("(~~+)", '', str(word)).lower()   #remove ~ if it occors more than one time consecutively
            word=re.sub("(\+\++)", '', str(word)).lower()   #remove + if it occors more than one time consecutively
            word=re.sub("(\.\.+)", '', str(word)).lower()   #remove . if it occors more than one time consecutively

            word=re.sub(r"[<>()|&©ø\[\]\'\",;?~*!’. ]", '', str(word)).lower() #remove <>()|&©ø"',;?~*!

            word=re.sub("(mailto:)", '', str(word)).lower() #remove mailto:
            word=re.sub(r"(\\x9\d)", '', str(word)).lower() #remove \x9* in text
            word=re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(word)).lower() #replace INC nums to INC_NUM
            word=re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', str(word)).lower() #replace CM# and CHG# to CM_NUM


            word=re.sub("(\.\s+)", '', str(word)).lower() #remove full stop at end of words(not between)
            word=re.sub("(\-\s+)", '', str(word)).lower() #remove - at end of words(not between)
            word=re.sub("(\:\s+)", '', str(word)).lower() #remove : at end of words(not between)
            
            word = re.sub(r"\d+'", '', str(word)).lower() #Remove numbers
            word = word.strip()
            

            word = re.sub("(\s+)",'',str(word)).lower() #remove multiple spaces

            #Should always be last
            word=re.sub("(\s+.\s+)", '', str(word)).lower() #remove any single charecters hanging between 2 spaces
            
            
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix

def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix

def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


# # Define Document

# In[3]:


#text = "The Daman and Diu administration on Wednesday withdrew a circular that asked women staff to tie rakhis on male colleagues after the order triggered a backlash from employees and was ripped apart on social media.The union territory?s administration was forced to retreat within 24 hours of issuing the circular that made it compulsory for its staff to celebrate Rakshabandhan at workplace.?It has been decided to celebrate the festival of Rakshabandhan on August 7. In this connection, all offices/ departments shall remain open and celebrate the festival collectively at a suitable time wherein all the lady staff shall tie rakhis to their colleagues,? the order, issued on August 1 by Gurpreet Singh, deputy secretary (personnel), had said.To ensure that no one skipped office, an attendance report was to be sent to the government the next evening.The two notifications ? one mandating the celebration of Rakshabandhan (left) and the other withdrawing the mandate (right) ? were issued by the Daman and Diu administration a day apart. The circular was withdrawn through a one-line order issued late in the evening by the UT?s department of personnel and administrative reforms.?The circular is ridiculous. There are sensitivities involved. How can the government dictate who I should tie rakhi to? We should maintain the professionalism of a workplace? an official told Hindustan Times earlier in the day. She refused to be identified.The notice was issued on Daman and Diu administrator and former Gujarat home minister Praful Kodabhai Patel?s direction, sources said.Rakshabandhan, a celebration of the bond between brothers and sisters, is one of several Hindu festivities and rituals that are no longer confined of private, family affairs but have become tools to push politic al ideologies.In 2014, the year BJP stormed to power at the Centre, Rashtriya Swayamsevak Sangh (RSS) chief Mohan Bhagwat said the festival had ?national significance? and should be celebrated widely ?to protect Hindu culture and live by the values enshrined in it?. The RSS is the ideological parent of the ruling BJP.Last year, women ministers in the Modi government went to the border areas to celebrate the festival with soldiers. A year before, all cabinet ministers were asked to go to their constituencies for the festival."

text = "The Daman and Diu administration on Wednesday withdrew a circular that asked women staff to tie rakhis on male colleagues after the order triggered a backlash from employees and was ripped apart on social media. The union territory’s administration was forced to retreat within 24 hours of issuing the circular that made it compulsory for its staff to celebrate Rakshabandhan at workplace. It has been decided to celebrate the festival of Rakshabandhan on August 7. In this connection, all offices/ departments shall remain open and celebrate the festival collectively at a suitable time wherein all the lady staff shall tie rakhis to their colleagues, the order, issued on August 1 by Gurpreet Singh, deputy secretary (personnel), had said. To ensure that no one skipped office, an attendance report was to be sent to the government the next evening. The two notifications one mandating the celebration of Rakshabandhan (left)and the other withdrawing the mandate (right) were issued by the Daman and Diu administration a day apart. The circular was withdrawn through a one-line order issued late in the evening by the UT’s department of personnel and administrative reforms. The circular is ridiculous. There are sensitivities involved. How can the government dictate who I should tie rakhi to we should maintain the professionalism of a workplace an official told Hindustan Times earlier in the day. She refused to be identified. The notice was issued on Daman and Diu administrator and former Gujarat home minister Praful Kodabhai Patel’s direction, sources said.Rakshabandhan, a celebration of the bond between brothers and sisters, is one of several Hindu festivities and rituals that are no longer confined of private, family affairs but have become tools to push politic al ideologies. In 2014, the year BJP stormed to power at the Centre, Rashtriya Swayamsevak Sangh (RSS) chief Mohan Bhagwat said the festival had ?national significance and should be celebrated widely to protect Hindu culture and live by the values enshrined in it. The RSS is the ideological parent of the ruling BJP. Last year, women ministers in the Modi government went to the border areas to celebrate the festival with soldiers. A year before, all cabinet ministers were asked to go to their constituencies for the festival"

'''
t1 = "The Daman and Diu administration on Wednesday withdrew a circular that asked women staff to tie rakhis on male colleagues after the order triggered a backlash from employees and was ripped apart on social media.The union territory?s administration was forced to retreat within 24 hours of issuing the circular that made it compulsory for its staff to celebrate Rakshabandhan at workplace.?It has been decided to celebrate the festival of Rakshabandhan on August 7. In this connection, all offices/ departments shall remain open and celebrate the festival collectively at a suitable time wherein all the lady staff shall tie rakhis to their colleagues,? the order, issued on August 1 by Gurpreet Singh, deputy secretary (personnel), had said.To ensure that no one skipped office, an attendance report was to be sent to the government the next evening.The two notifications ? one mandating the celebration of Rakshabandhan (left) and the other withdrawing the mandate (right) ? were issued by the Daman and Diu administration a day apart. The circular was withdrawn through a one-line order issued late in the evening by the UT?s department of personnel and administrative reforms.?The circular is ridiculous. There are sensitivities involved. How can the government dictate who I should tie rakhi to? We should maintain the professionalism of a workplace? an official told Hindustan Times earlier in the day. She refused to be identified.The notice was issued on Daman and Diu administrator and former Gujarat home minister Praful Kodabhai Patel?s direction, sources said.Rakshabandhan, a celebration of the bond between brothers and sisters, is one of several Hindu festivities and rituals that are no longer confined of private, family affairs but have become tools to push politic al ideologies.In 2014, the year BJP stormed to power at the Centre, Rashtriya Swayamsevak Sangh (RSS) chief Mohan Bhagwat said the festival had ?national significance? and should be celebrated widely ?to protect Hindu culture and live by the values enshrined in it?. The RSS is the ideological parent of the ruling BJP.Last year, women ministers in the Modi government went to the border areas to celebrate the festival with soldiers. A year before, all cabinet ministers were asked to go to their constituencies for the festival."
t2 = "From her special numbers to TV?appearances, Bollywood actor Malaika Arora Khan has managed to carve her own identity. The actor, who made her debut in the Hindi film industry with the blockbuster debut opposite Shah Rukh Khan in Chaiyya Chaiyya from Dil Se (1998), is still remembered for the song. However, for trolls, she is a woman first and what matters right now is that she divorced a ?rich man?.  On Wednesday, Malaika Arora shared a gorgeous picture of herself on Instagram and a follower decided to troll her for using her ?alumni? (read alimony) money to wear ?short clothes and going to gym or salon?. Little did he/she know that the Munni Badnam star would reply with the perfect comeback. Take a look at the interaction:     Super excited to be affiliated with Khanna Jewellers @khannajewellerskj as their brand ambassador. Crafted to perfection, their stunning statement jewellery is a must have for every jewellery lover. #khannajewellers...#maksquad?? #hair @hairbypriyanka #stylist @manekaharisinghani #manager @ektakauroberoi #mua? @subbu28 #photographer @prasdnaik A post shared by Malaika Arora Khan (@malaikaarorakhanofficial) on Aug 2, 2017 at 6:20am PDT Then, Malaika decided to reply: The entire conversation only proves that no matter if a woman is successful, she will be attacked the moment she decides to step out of bounds the society decided for her. Apart from being a successful woman who lives life on her own terms, Malaika has literally played all the roles traditionally prescribed for a woman - she married quite early, had a son and raised him and was always around with the ?khandan?. But then, she got divorced and alimony is the taunt being thrown at her. The details of the alimony are only known to Malaika, her husband Arbaaz Khan and perhaps the family. The couple has handled the divorce with the utmost dignity. But we can vouch for the fact that she did not  need an alimony to buy clothes (short or not, her choice), go on vacations and enjoy her life. If anything, she is as successful, if not more, than her ex-husband.What happened between Arbaaz and Malaika is their personal concern. But to claim that Malaika married and then divorced Arbaaz for money doesn?t hold water. For those who do not agree, please get a course in feminism and for others, here?s a playlist of some of her most popular songs. Follow @htshowbiz for more"
t3 = "The Indira Gandhi Institute of Medical Sciences (IGIMS) in Patna amended its marital declaration form on Thursday, replacing the word ?virgin? with ?unmarried? after controversy.Until now, new recruits to the super-specialty medical institute in the state capital were required to declare if they were bachelors, widowers or virgins.IGIMS medical superintendent Dr Manish Mandal said institute director Dr NR Biswas held a meeting on Thursday morning before directing that the word ?virgin? on the marital declaration form be immediately replaced with ?unmarried?. Dr Biswas had just returned after a four-day leave of absence.Earlier, Bihar health minister Mangal Pandey had ended up redefining the very meaning of virginity in his attempts to justify the awkward phrasing of the question in the form. Following a public furore over the document on Wednesday, the minister told news channels that there was nothing wrong with using the word ?virgin? because it simply meant ?kanya? or ?kunwari? ? which means an unmarried girl.Pandey had joined the cabinet just three days ago.Sources said the chief minister?s office had also taken cognizance of the issue, and asked for a copy of the form. It had even asked why the question was introduced in the first place.In its response, the management of the autonomous super-specialty health facility had clarified on Wednesday that it was in adherence to the central civil services rules followed by the All India Institute of Medical Sciences in New Delhi.The previous version of the marital declaration form, which purportedly asked new recruits if they were virgins. The marital declaration form had been in existence since the inception of the institute in 1983. Some officials blamed the faux pas on poor translation on the part of individuals who drafted the document.?The word ?virgin? mentioned on the form had nothing to do with the virginity of any employee. It only sought to know the employees? marital status, so their dues could be settled on the basis of their declaration in the event of death while in service,? said Dr Mandal."
t4 = "Lashkar-e-Taiba's Kashmir commander Abu Dujana was killed in an encounter in a village in Pulwama district of Jammu and Kashmir earlier this week. Dujana, who had managed to give the security forces a slip several times in the past, carried a bounty of Rs 15 lakh on his head.Reports say that Dujana had come to meet his wife when he was trapped inside a house in Hakripora village. Security officials involved in the encounter tried their best to convince Dujana to surrender but he refused, reports say.According to reports, Dujana rejected call for surrender from an Army officer. The Army had commissioned a local to start a telephonic conversation with Dujana. After initiating the talk, the local villager handed over the phone to the army officer.\"Kya haal hai? Maine kaha, kya haal hai (How are you. I asked, how are you)?\" Dujana is heard asking the officer. The officer replies: \"Humara haal chhor Dujana. Surrender kyun nahi kar deta. Tu galat kar rha hai (Why don't you surrender? You have married this girl. What you are doing isn't right.)\"When told that he is being used by Pakistani agencies as a pawn, Dujana, who sounded calm and unperturbed of the situation, said \"Hum nikley they shaheed hone. Main kya karu. Jisko game khelna hai, khelo. Kabhi hum aage, kabhi aap, aaj aapne pakad liya, mubarak ho aapko. Jisko jo karna hai karlo (I had left home for martyrdom. What can I do? Today you caught me. Congratulations. \"Surrender nahi kar sakta. Jo meri kismat may likha hoga, Allah wahi karega, theek hai? (I won't surrender. Allaah would do whatever is there in my fate)\" Dujana went on to say. Dujana, who belonged to Pakistan, was Lashkar-e-Taiba's divisional commander in south Kashmir. He was among the top 10 terrorists identified by the Indian Army in Jammu and Kashmir.With a Rs 15 lakh bounty on his head, Dujana was labelled an 'A++' terrorist - the top grade which was also given to Burhan Wani.Security forces received inputs that during the last few days he was frequenting the houses of his wife Rukaiya and girlfriend Shazia. Police was keeping a watch on both the houses. when it was confirmed he was present in his wife's house, security forces moved in to trap him.ALSO READ:After Abu Dujana, security forces prepare new hitlist of most wanted terroristsAbu Dujana encounter: Jilted lover turned police informer led security forces to LeT commander"
t5 = "Hotels in Mumbai and other Indian cities are to train their staff to spot signs of sex trafficking such as frequent requests for bed linen changes or a \"Do not disturb\" sign left on the door for days on end. The group behind the initiative is also developing a mobile phone app - Rescue Me - which hotel staff can use to alert local police and senior anti-trafficking officers if they see suspicious behavior. \"Hotels are breeding grounds for human trade,\" said Sanee Awsarmmel, chairman of the alumni group of Maharashtra State Institute of Hotel Management and Catering Technology. \"(We) have hospitality professionals working in hotels across the country. We are committed to this cause.\"The initiative, spearheaded by the alumni group and backed by the Maharashtra state government, comes amid growing international recognition that hotels have a key role to play in fighting modern day slavery. MAHARASHTRA MAJOR DESTINATION FOR TRAFFICKED GIRLS Maharashtra, of which Mumbai is the capital, is a major destination for trafficked girls who are lured from poor states and nearby countries on the promise of jobs, but then sold into the sex trade or domestic servitude. With rising property prices, some traditional red light districts like those in Mumbai have started to disappear pushing the sex trade underground into private lodges and hotels, which makes it hard for police to monitor.Awsarmmel said hotels would be told about 50 signs that staff needed to watch out for.These include requests for rooms with a view of the car park which are favored by traffickers as they allow them to vet clients for signs of trouble and check out their cars to gauge how much to charge.Awsarmmel said hotel staff often noticed strange behavior such as a girl's reticence during the check-in process or her dependence on the person accompanying her to answer questions and provide her proof of identity.But in most cases, staff ignore these signs or have no idea what to do, he told the Thomson Reuters Foundation.RESCUE ME APP The Rescue Me app - to be launched in a couple of months - will have a text feature where hotel staff can fill in details including room numbers to send an alert to police.Human trafficking is the world's fastest growing criminal enterprise worth an estimated $150 billion a year, according to the International Labor Organization, which says nearly 21 million people globally are victims of forced labor and trafficking.Last year, major hotel groups, including the Hilton and Shiva Hotels, pledged to examine their supply chains for forced labor, and train staff how to spot and report signs of trafficking.Earlier this year, Mexico City also launched an initiative to train hotel staff about trafficking.Vijaya Rahatkar, chairwoman of the Maharashtra State Women's Commission, said the initiative would have an impact beyond the state as the alumni group had contact with about a million small hotels across India.The group is also developing a training module on trafficking for hotel staff and hospitality students which could be used across the country.ALSO READFYI | Legal revenge: Child sex trafficking survivors get 'School of Justice' to fight their own battlesMumbai: Woman DJ arrested in high-profile sex racket case"
'''




# # Calling the functions defined above for matrix creation

# In[4]:



# 1 Sentence Tokenize
sentences = sent_tokenize(text)
total_documents = len(sentences)
print(sentences)
print(total_documents, end="\n")

print("Step 2: **********************Create the Frequency matrix of the words in each sentence.*********************", end="\n")

# 2 Create the Frequency matrix of the words in each sentence.
freq_matrix = _create_frequency_matrix(sentences)
print(freq_matrix)

print("Step 3: *******************Calculate TermFrequency and generate a matrix****************************", end="\n")

# 3 Calculate TermFrequency and generate a matrix
tf_matrix = _create_tf_matrix(freq_matrix)
print(tf_matrix)

print("Step 4: ****************creating table for documents per words************************************", end="\n")

# 4 creating table for documents per words
count_doc_per_words = _create_documents_per_words(freq_matrix)
print(count_doc_per_words)

print("Step 5: ***********************Calculate IDF and generate a matrix****************************", end="\n")

# 5 Calculate IDF and generate a matrix
idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
print(idf_matrix)

print("Step 6: ***********************Calculate TF-IDF and generate a matrix****************************", end="\n")
# 6 Calculate TF-IDF and generate a matrix
tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
print(tf_idf_matrix)

print("Step 7: **********************Important Algorithm: score the sentences**************************", end="\n")
# 7 Important Algorithm: score the sentences
sentence_scores = _score_sentences(tf_idf_matrix)
print(sentence_scores)


# # Apply Clustering on Sentences

# # Data preparation before applying KMeans Clustering

# In[5]:


#Creating a dataframe from tfidfmatrix for data processing 
df = pd.DataFrame(tf_idf_matrix)

#Transaposing the dataframe so that the sentences become rows
df = df.T

#Replacing NAN with ZERO 
df.replace(np.nan,0,inplace=True)

#Storing the dataframe in array and assign it to "x"
x = df.values


# In[6]:


x.shape


# # Elbow plot to identify true clusters

# In[7]:


# Run the Kmeans algorithm and get the index of data points clusters
sse = []
list_k = list(range(1, 15))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(x)
    sse.append(km.inertia_)
    print("Number of cluster2: {}".format(k), end=" ")
    print("Value of SSE/Distortion: {}".format(km.inertia_))

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');


# # Apply clustering on the data using sklearn KMeans

# In[8]:


kmeans = cluster.KMeans(n_clusters=8, init='k-means++',
                       max_iter=100, n_init=1, verbose=0, random_state=3425)
            
kmeans.fit(x)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_


# In[9]:


labels


# In[10]:


centroids.shape


# # Getting Cluster with highest Word frequency and then summarization

# # Creating Dataframe of labels

# In[11]:


lbl = pd.DataFrame(labels, columns=["label"])


# # Creating a copy of dataframe of sentences

# In[12]:


# Creating a copy of dataframe of sentences
final_df = df

#Setting a Cluster column in the dataframe of sentences
final_df['cluster'] = lbl.values

#Calculating null columns 
test = (final_df == 0).astype(int).sum(axis=1)

#Creating a datframe
test_df = pd.DataFrame(test)

#Adding more columns to dataframe
test_df['total'] = 163
test_df['not_null'] = test_df['total'] - test_df[0]

#Now adding cluster column to it
test_df['cluster'] = final_df['cluster']

test_df.rename(columns={0:"blank_columns"}, inplace=True)


#Steps to identify high word frequency cluster
x = pd.DataFrame(test_df.groupby('cluster')['not_null'].sum())
x.rename(columns={"not_null":"Occurrence"}, inplace=True)
print(x)


y = pd.DataFrame(test_df.groupby('cluster')['not_null'].count())
y.rename(columns={"not_null":"Not_null_rows"}, inplace=True)
print(y)

#Concatenating x and y
frequency_df_temp = pd.concat([x , y], axis=1)

#Creating a Frequency column
frequency_df_temp["frequency"] = frequency_df_temp["Occurrence"]/(frequency_df_temp["Not_null_rows"]*163)
print(frequency_df_temp)

#Fetching cluster with highest frequency
frequency_df_temp[frequency_df_temp["frequency"] == max(frequency_df_temp["frequency"])].index[0]

#Subsetting only sentences which lies in the high word frequency cluster
#Creating a datarframe of only rows which lies in the above cluster
df1 = final_df[final_df.cluster == frequency_df_temp[frequency_df_temp["frequency"] == max(frequency_df_temp["frequency"])].index[0]]

#Creating dictionary of dataframe 
final_dict = df1.set_index(df1.T.columns).T.to_dict('list')

#Original sentences score dictionary
print(sentence_scores)

#Fetching matching sentences
l = {}
for i in final_dict.keys():
    for j in sentence_scores.keys():
        if i == j:
            l[i] = 1

            
#Function to generate summary
def _generate_summary(sentences, sentenceValue):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue:
            summary += " " + sentence
            sentence_count += 1

    return summary

#Generate summary
_generate_summary(sentences, l)

summary_text = _generate_summary(sentences, l)

#Print SUMMARY of text
print(summary_text)


# # BLEU Score comparison

# In[13]:


# n-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
text =  word_tokenize(text)
reference = [text]
summary_text = word_tokenize(summary_text)
candidate = summary_text
print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))


# In[14]:


# 4-gram cumulative BLEU
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)


# In[ ]:





# # Using Silhoutte Method

# In[15]:


from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


# In[16]:


#Creating a dataframe from tfidfmatrix for data processing 
df = pd.DataFrame(tf_idf_matrix)

#Transaposing the dataframe so that the sentences become rows
df = df.T

#Replacing NAN with ZERO 
df.replace(np.nan,0,inplace=True)

#Storing the dataframe in array and assign it to "x"
x = df.values


# In[17]:


X=x


# In[18]:


X


# In[19]:



range_n_clusters = [2, 3, 4, 5, 6]
max_score = 0
best_cluster_value = 0
best_cluster_label = 0

for n_clusters in range_n_clusters:
    
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    
    print("Cluster lables for clusters = {} are {}".format(n_clusters, cluster_labels))

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    
    #Check best score and labels
    if max_score < silhouette_avg:
        max_score = silhouette_avg
        best_cluster_value = n_clusters
        best_cluster_label = cluster_labels
    
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()


# In[20]:


print("Max score is: ",max_score)
print("best_cluster_value is ",best_cluster_value)
print("best_cluster_label is ",best_cluster_label)


# # Summarization using Silhoutte

# In[21]:


lbl = pd.DataFrame(best_cluster_label , columns=["label"])

# In[63]:
lbl

# In[64]:
final_df = df

# In[65]:
final_df

# In[66]:
#Setting a Cluster column in the dataframe of sentences
final_df['cluster'] = lbl.values

# In[67]:
final_df

# In[68]:
#Calculating null columns 
test = (final_df == 0).astype(int).sum(axis=1)

# In[69]:
#Creating a datframe
test_df = pd.DataFrame(test)

# In[70]:
#Adding more columns to dataframe
test_df['total'] = 163
test_df['not_null'] = test_df['total'] - test_df[0]

# In[71]:
#Now adding cluster column to it
test_df['cluster'] = final_df['cluster']

# In[72]:
test_df.rename(columns={0:"blank_columns"}, inplace=True)

# In[73]:
test_df.columns

# In[74]:
test_df

# # Steps to identify high word frequency cluster
# In[75]:
x = pd.DataFrame(test_df.groupby('cluster')['not_null'].sum())
x.rename(columns={"not_null":"Occurrence"}, inplace=True)
print(x)

# In[76]:
y = pd.DataFrame(test_df.groupby('cluster')['not_null'].count())
y.rename(columns={"not_null":"Not_null_rows"}, inplace=True)
print(y)

# In[77]:
frequency_df_temp = pd.concat([x , y], axis=1)

# In[78]:
frequency_df_temp["frequency"] = frequency_df_temp["Occurrence"]/(frequency_df_temp["Not_null_rows"]*163)

# In[79]:
frequency_df_temp

# In[80]:
#Fetching cluster with highest frequency
frequency_df_temp[frequency_df_temp["frequency"] == max(frequency_df_temp["frequency"])].index[0]

# # Subsetting only sentences which lies in the high word frequency cluster
# In[81]:
#Creating a datarframe of only rows which lies in the above cluster
df1 = final_df[final_df.cluster == frequency_df_temp[frequency_df_temp["frequency"] == max(frequency_df_temp["frequency"])].index[0]]

# In[82]:
#Creating dictionary of dataframe 
final_dict = df1.set_index(df1.T.columns).T.to_dict('list')

# In[83]:
final_dict

# In[84]:
#Original sentences score dictionary
sentence_scores

# In[85]:
#Fetching matching sentences
l = {}
for i in final_dict.keys():
    for j in sentence_scores.keys():
        if i == j:
            l[i] = 1

# In[86]:
l

# In[87]:
#Function to generate summary
def _generate_summary(sentences, sentenceValue):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue:
            summary += " " + sentence
            sentence_count += 1

    return summary

# In[135]:
_generate_summary(sentences, l)
#As randomness is a factor in the K-Means algorithm,the results reported were taken as the average of 100 separate instances of the clustering algorithm

# In[136]:
summary_text = _generate_summary(sentences, l)

# In[41]:
summary_text


# In[22]:


text


# # BLEU SCORE COMPARISON

# In[23]:


# n-gram individual BLEU
#text =  word_tokenize(text)
reference = [text]
summary_text = word_tokenize(summary_text)
candidate = summary_text
print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))


# In[24]:


score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)


# In[ ]:





# # GAPS STATISTICS

# In[25]:


#https://anaconda.org/milesgranger/gap-statistic/notebook


# # Implementation of the Gap Statistic with some help from SciKit Learn in calculating the dispersion

# In[26]:


def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for 
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


# In[27]:


#Creating a dataframe from tfidfmatrix for data processing 
df = pd.DataFrame(tf_idf_matrix)

#Transaposing the dataframe so that the sentences become rows
df = df.T

#Replacing NAN with ZERO 
df.replace(np.nan,0,inplace=True)

#Storing the dataframe in array and assign it to "x"
x = df.values


# In[28]:


k, gapdf = optimalK(x, nrefs=5, maxClusters=15)
print('Optimal k is: ', k)


# In[29]:


plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()


# In[30]:


km = KMeans(k)
km.fit(x)


# In[31]:


km.labels_


# In[32]:


x


# In[33]:


lbl = pd.DataFrame(km.labels_ , columns=["label"])

# In[63]:
lbl

# In[64]:
final_df = df

# In[65]:
final_df

# In[66]:
#Setting a Cluster column in the dataframe of sentences
final_df['cluster'] = lbl.values

# In[67]:
final_df

# In[68]:
#Calculating null columns 
test = (final_df == 0).astype(int).sum(axis=1)

# In[69]:
#Creating a datframe
test_df = pd.DataFrame(test)

# In[70]:
#Adding more columns to dataframe
test_df['total'] = 163
test_df['not_null'] = test_df['total'] - test_df[0]

# In[71]:
#Now adding cluster column to it
test_df['cluster'] = final_df['cluster']

# In[72]:
test_df.rename(columns={0:"blank_columns"}, inplace=True)

# In[73]:
test_df.columns

# In[74]:
test_df

# # Steps to identify high word frequency cluster
# In[75]:
x = pd.DataFrame(test_df.groupby('cluster')['not_null'].sum())
x.rename(columns={"not_null":"Occurrence"}, inplace=True)
print(x)

# In[76]:
y = pd.DataFrame(test_df.groupby('cluster')['not_null'].count())
y.rename(columns={"not_null":"Not_null_rows"}, inplace=True)
print(y)

# In[77]:
frequency_df_temp = pd.concat([x , y], axis=1)

# In[78]:
frequency_df_temp["frequency"] = frequency_df_temp["Occurrence"]/(frequency_df_temp["Not_null_rows"]*163)

# In[79]:
frequency_df_temp

# In[80]:
#Fetching cluster with highest frequency
frequency_df_temp[frequency_df_temp["frequency"] == max(frequency_df_temp["frequency"])].index[0]

# # Subsetting only sentences which lies in the high word frequency cluster
# In[81]:
#Creating a datarframe of only rows which lies in the above cluster
df1 = final_df[final_df.cluster == frequency_df_temp[frequency_df_temp["frequency"] == max(frequency_df_temp["frequency"])].index[0]]

# In[82]:
#Creating dictionary of dataframe 
final_dict = df1.set_index(df1.T.columns).T.to_dict('list')

# In[83]:
final_dict

# In[84]:
#Original sentences score dictionary
sentence_scores

# In[85]:
#Fetching matching sentences
l = {}
for i in final_dict.keys():
    for j in sentence_scores.keys():
        if i == j:
            l[i] = 1

# In[86]:
l

# In[87]:
#Function to generate summary
def _generate_summary(sentences, sentenceValue):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue:
            summary += " " + sentence
            sentence_count += 1

    return summary

# In[135]:
_generate_summary(sentences, l)
#As randomness is a factor in the K-Means algorithm,the results reported were taken as the average of 100 separate instances of the clustering algorithm

# In[136]:
summary_text = _generate_summary(sentences, l)

# In[41]:
summary_text


# In[34]:


# n-gram individual BLEU
#text =  word_tokenize(text)
reference = [text]
summary_text = word_tokenize(summary_text)
candidate = summary_text
print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))


# In[35]:


# 4-gram cumulative BLEU
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)


# In[ ]:




