import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier

## Page expands to full width
st.set_page_config(layout="wide")

# Title
st.title('Multidimensional Poverty Predictor')

# Description
st.markdown('''
What drives poverty in Spain? Find out about the relationship between poverty and other variables among the Spanish adult population. This app allows you to explore poverty statistics in depth. You can try the poverty predictor model as well!\n
The data for this project was retrieved from the 2019 Income and Living Conditions survey conducted by the INE
(*Instituto Nacional de Estadistica*), Spain's official agency for statistical services.\n
* **Python libraries**: streamlit, pandas, numpy, matplotlib, seaborn, sklearn
* **Data Source**: [ine.es](https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176807&menu=ultiDatos&idp=1254735976608)\n
Access to GitHub repo [here](https://github.com/deividvalerius/Multidimensional-Poverty-Predictor)
''')

#Loading data from Git Repo
url = 'https://raw.githubusercontent.com/deividvalerius/Multidimensional-Poverty-Predictor/master/Data/variables.csv'
df = pd.read_csv(url)

#Creating sidebar for interactive user imputation
st.sidebar.header('Select variables to filter')

#Sidebar - Sex
sex = sorted(df.sex.unique())
selected_sex = st.sidebar.multiselect('Sex', sex, sex)

#Sidebar - Civil status
civil_status = sorted(df.civil_status.unique())
selected_civil_status = st.sidebar.multiselect('Civil status', civil_status, civil_status)

#Sidebar - Familial status
familial_status = sorted(df.familial_status.unique())
selected_familial_status = st.sidebar.multiselect('Familial status', familial_status, familial_status)

#Sidebar - Region
region = sorted(df.region.unique())
selected_region = st.sidebar.multiselect('Region', region, region)

#Sidebar - Population density
population_density = sorted(df.population_density.unique())
selected_population_density = st.sidebar.multiselect('Population density', population_density, population_density)

#Sidebar - Citizenship
citizenship = sorted(df.citizenship.unique())
selected_citizenship = st.sidebar.multiselect('Citizenship', citizenship, citizenship)

#Sidebar - Tenure status
tenure_status = sorted(df.tenure_status.unique())
selected_tenure_status = st.sidebar.multiselect('Tenure status', tenure_status, tenure_status)

#Sidebar - Education level
education_level = sorted(df.education_level.unique())
selected_education_level = st.sidebar.multiselect('Education level', education_level, education_level)

#Sidebar - Working status
working_status = sorted(df.working_status.unique())
selected_working_status = st.sidebar.multiselect('Working status', working_status, working_status)

#Sidebar - Occupation
occupation = sorted(df.occupation.unique())
selected_occupation = st.sidebar.multiselect('Occupation', occupation, occupation)

#Sidebar - Bad health
bad_health = sorted(df.bad_health.unique())
selected_bad_health = st.sidebar.multiselect('Bad health', bad_health, bad_health)

#Sidebar - Material deprivation
material_deprivation = sorted(df.material_deprivation.unique())
selected_material_deprivation = st.sidebar.multiselect('Material deprivation', material_deprivation, material_deprivation)

#Sidebar - Age
selected_age = st.sidebar.slider('Age', int(df.age.min()), int(df.age.max()), (int(df.age.min()), int(df.age.max())))

#Sidebar - Years worked
selected_years_worked = st.sidebar.slider('Years worked', int(df.years_worked.min()), int(df.years_worked.max()), (int(df.years_worked.min()), int(df.years_worked.max())))

#Sidebar - Hours a week worked
selected_hours_week_worked = st.sidebar.slider('Hours a week worked', int(df.hours_week_worked.min()), int(df.hours_week_worked.max()), (int(df.hours_week_worked.min()), int(df.hours_week_worked.max())))

#Sidebar - Adjusted income
selected_adjusted_income = st.sidebar.slider('Adjusted income', -20000, 180000, (-20000, 180000), 1000)

#Sidebar - Proportion of social welfare
selected_proportion_social_welfare = st.sidebar.slider('Proportion of social welfare', 0.0, 1.0, (0.0, 1.0), 0.01)

#Putting together selected data frame
selected_df = df[df.sex.isin(selected_sex)
            & df.civil_status.isin(selected_civil_status)
            & df.familial_status.isin(selected_familial_status)
            & df.region.isin(selected_region)
            & df.population_density.isin(selected_population_density)
            & df.citizenship.isin(selected_citizenship)
            & df.tenure_status.isin(selected_tenure_status)
            & df.education_level.isin(selected_education_level)
            & df.working_status.isin(selected_working_status)
            & df.occupation.isin(selected_occupation)
            & df.material_deprivation.isin(selected_material_deprivation)
            & df.age.between(selected_age[0], selected_age[1])
            & df.years_worked.between(selected_years_worked[0], selected_years_worked[1])
            & df.hours_week_worked.between(selected_hours_week_worked[0], selected_hours_week_worked[1])
            & df.adjusted_income.between(selected_adjusted_income[0], selected_adjusted_income[1])
            & df.proportion_social_welfare.between(selected_proportion_social_welfare[0], selected_proportion_social_welfare[1])].reset_index(drop=True)

#Explanatory text
mat_description = st.beta_expander("What is multidimensional poverty?")
mat_description.write('''
**Material deprivation**: Is the target variable of this project. Is a complex *multidimensional* variable based on nine indicators associated with living in poverty. These are:\n
- Inability to afford paying for one week annual holiday away from home.\n
- Inability to afford a meal with meat, chicken or fish (or vegetarian equivalent) every second day.\n
- Inability to keep home adequately warm.\n
- Inability to face unexpected expenses.\n
- Arrears on utility bills, mortgage or rental payments, or hire purchases or other loan payments.\n
- Inability to afford a car.\n
- Inability to afford a telephone.\n
- Inability to afford a TV.\n
- Inability to afford a washing machine.\n
Households with positive values on at least four elements of this list are classified with material deprivation.
''')
var_description = st.beta_expander("Get to know the other variables")
var_description.write('''
**Sex**: In the binary physical sense. The source data does not cover gender identity yet.\n
**Age**: From 18 onwards. The source data assign the same age to those older than 85, hence the histogram shape.\n
**Civil Status**: Married 'de facto' accounts for non-married co-habitants partners.\n
**Familial Status**: Whether underage individuals are part of a household or not.\n
**Region**: Autonomous Communities of Spain.\n
**Population density**: Densely equals population 50000+ and 1500 inhabitants/km2 nearby. Intermediate equals population 5000+ and 300 inhabitants/km2 nearby.\n
**Citizenship**: Those Spanish citizens born elsewhere have the label 'Spain (naturalized).\n
**Tenure status** : Over the household accommodation. 'Tenancy at reduced rate' accounts for special cases when rent reduction takes place like social housing.\n
**Education level**: Highest recognized educational certification.\n
**Working status**: Main activity status as considered by the respondent.\n
**Occupation**: Job classification inside the International Standard Classification of Occupation major groups.\n
**Years worked**: Total number of years in employment.\n
**Hours a week worked**: Total number of hours a week work on average in any job.\n
**Adjusted income**: Annual household income divided by unit of consumption (1 + (other people older than 14 x 0.5) + (other people 14 or younger x 0.3)). *Intuitively*: Annual income to maintain lifestyle if living alone.\n
**Proportion of social welfare**: Proportion of annual income coming from welfare benefits.
''')

#Subtitle for the variable options
st.subheader('Choose a variable for display')

#Selectbox for the X and Y variable
variable_names = {c[0].upper() + c[1:].lower().replace('_', ' '): c for c in df.columns} #editing string format from the df column names
selectbox_options = list(variable_names.keys())[1:]
x = st.selectbox('X', selectbox_options, index=selectbox_options.index('Sex'))
y = st.selectbox('Y', selectbox_options)

#Only X variable option
one_variable_button = st.button('Select only the X variable')

#Defining functions - See reference in my_functions.ipynb
def weighted_freq(data, cat_column):
    dummy = pd.get_dummies(cat_column)
    for c in dummy.columns:
        dummy['weight_' + str(c)] = dummy[c] * data.weight
    freq_dict = {}
    for c in dummy.columns:
        if str(c)[0] == 'w':
            freq_dict[c[7:]] = dummy[c].sum() / data.weight.sum()
    return freq_dict

def barplot(x=variable_names[x], data=selected_df, legend=True):
    fig, ax = plt.subplots()
    labels = sorted(data[x].unique())
    label_colors = ['C'+str(i) for i in range(len(labels))]
    x_ticks = range(len(labels))
    freq = weighted_freq(data, data[x])
    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.5))
    ax = plt.subplot()
    ax.bar(x_ticks, [freq[label] for label in labels], color=label_colors)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels)
    plt.xlabel(x)
    plt.ylabel('Frequency')
    if legend==False:
        ax.set_xticklabels(labels)
    else:
        ax.set_xticklabels(['' for label in labels])
        for i in range(len(labels)):
            ax.bar([0], [0], label=labels[i][:40], color=label_colors[i])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    col2.pyplot(fig)

def histplot(x=variable_names[x], data=selected_df, bins=68):
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.5))
    ax = sns.histplot(x=data[x], data=data, weights='weight', bins=bins, stat='density')
    col2.pyplot(fig)

def length_calculator(labels):
    a = 4
    multiplier = 1
    rows = 2
    for i in range(labels):
        if i+1 >= a + 3:
            a += 3
            multiplier += 0.5
            rows += 1
    return multiplier, rows

def bardiagram(x=variable_names[x], y=variable_names[y], data=selected_df, legend=True):
    fig, ax = plt.subplots()
    titles = sorted(data[x].unique())
    labels = sorted(data[y].unique())
    label_colors = ['C'+str(i) for i in range(len(labels))]
    x_ticks = range(len(labels))
    length_multiplier, subplot_rows = length_calculator(len(titles))
    fig = plt.figure(figsize=(6.4*2, 4.8*2*length_multiplier))
    subplot = 1
    ytick_max_list = []
    for title in titles:
        subplot_data = data[data[x] == title]
        ytick_max = max(weighted_freq(subplot_data, subplot_data[y]).values())
        ytick_max_list.append(ytick_max)
    ytick_max_length = max(ytick_max_list)
    for title in titles:
        subplot_data = data[data[x] == title]
        freq = weighted_freq(subplot_data, subplot_data[y])
        ax = plt.subplot(subplot_rows, 3, subplot)
        ax.title.set_text(title[:40])
        ax.bar(x_ticks, [freq[label] if label in freq.keys() else 0 for label in labels], color=label_colors)
        ax.set_yticks([i/10 for i in range(int(ytick_max_length * 10)+2)])
        ax.set_xticks(x_ticks)
        if legend==False:
            ax.set_xticklabels(labels)
        else:
            ax.set_xticklabels(['' for label in labels])
        subplot += 1
    if legend==True:
        for i in range(len(labels)):
            ax.bar([0], [0], label=labels[i][:40], color=label_colors[i])
            ax.set_yticks([i/10 for i in range(int(ytick_max_length * 10)+2)])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    col2.pyplot(fig)

def weighted_cat(data, cat_column):
    dummy = pd.get_dummies(cat_column)
    for c in dummy.columns:
        dummy['weight_' + str(c)] = dummy[c] * data.weight
    weighted_cat = []
    for i in range(len(data)):
        for c in dummy.columns:
            if c[0] == 'w':
                for x in range(int(round(dummy[c][i]/1000, 0))):
                       weighted_cat.append(c[7:])
    return weighted_cat

def weighted_num(data, num_column):
    weighted_num = []
    for i in range(len(data)):
        for x in range(int(round(data.weight[i]/1000, 0))):
            weighted_num.append(num_column[i])
    return weighted_num

def boxplot(x=variable_names[x], y=variable_names[y], data=selected_df, legend=True):
    fig, ax = plt.subplots()
    weighted_data = pd.DataFrame({x: weighted_cat(data, data[x]), y: weighted_num(data, data[y])})
    labels = sorted(data[x].unique())
    palette = {labels[i]: 'C'+str(i) for i in range(len(labels))}
    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.5))
    ax = plt.subplot()
    boxplot = sns.boxplot(x=x, y=y, data=weighted_data, order=sorted(data[x].unique()), palette=palette)
    if legend == True:
        boxplot.set(xticklabels=[])
        for i in range(len(labels)):
            ax.bar([0], [0], label=labels[i][:40], color=palette[labels[i]])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    col2.pyplot(fig)

def scatterplot(x=variable_names[x], y=variable_names[y], data=selected_df):
    fig, ax = plt.subplots()
    weighted_data = pd.DataFrame({x: weighted_num(data, data[x]), y: weighted_num(data, data[y])})
    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.5))
    ax = sns.scatterplot(x=x, y=y, data=weighted_data)
    col2.pyplot(fig)

#Showing graph
with st.beta_container():
    col1, col2, col3 = st.beta_columns((1, 3, 1))
    if one_variable_button == False:
        if str(selected_df[variable_names[x]].dtype) == 'object':
            if str(selected_df[variable_names[y]].dtype) == 'object':
                bardiagram()
            else:
                boxplot()
        else:
            if str(selected_df[variable_names[y]].dtype) == 'object':
                st.write('Please swap the X and Y variables!')
            else:
                scatterplot()

    else:
        if str(selected_df[variable_names[x]].dtype) == 'object':
            barplot()
        else:
            bins = st.slider('bins', 1, 100, 68)
            histplot(bins=bins)

#Showing statistics
stats = st.beta_expander("Descriptive statistics")
if one_variable_button == False:
    if str(selected_df[variable_names[x]].dtype) == 'object':
        if str(selected_df[variable_names[y]].dtype) == 'object':
            for value in sorted(selected_df[variable_names[x]].unique()):
                segmented_df = selected_df[selected_df[variable_names[x]] == value].reset_index()
                freq_dict = weighted_freq(segmented_df, segmented_df[variable_names[y]])
                string_list = []
                for key in freq_dict:
                    string_list.append(f'**{key}**: {round((freq_dict[key] * 100), 2)}%')
                stats.write(f"***{value}*** --> {' '.join(string_list)}")

        else:
            for value in sorted(selected_df[variable_names[x]].unique()):
                segmented_df = selected_df[selected_df[variable_names[x]] == value].reset_index()
                weighted_var = pd.Series(weighted_num(segmented_df, segmented_df[variable_names[y]]))
                stats.write(f"***{value}*** --> **Mean**: {round(weighted_var.mean(), 2)}, **StDv**: {round(weighted_var.std(), 2)}, **Min**: {round(weighted_var.min(), 2)}, **1stQ**: {round(weighted_var.quantile(0.25), 2)}, **Median**: {round(weighted_var.median(), 2)}, **3stQ**: {round(weighted_var.quantile(0.75), 2)}, **Max**: {round(weighted_var.max(), 2)}")
    else:
        if str(selected_df[variable_names[y]].dtype) == 'object':
            pass
        else:
            scaler = MinMaxScaler()
            weighted_df = pd.DataFrame()
            weighted_df['x'] = weighted_num(selected_df, selected_df[variable_names[x]])
            weighted_df['y'] = weighted_num(selected_df, selected_df[variable_names[y]])
            weighted_df = pd.DataFrame(scaler.fit_transform(weighted_df))
            coeff = float(weighted_df.corr()[0][1])
            stats.write(f'**Pearson coefficient**: {round(coeff, 2)}')

else:
    if str(selected_df[variable_names[x]].dtype) == 'object':
        freq_dict = weighted_freq(selected_df, selected_df[variable_names[x]])
        string_list = []
        for key in freq_dict:
            string_list.append(f'**{key}**: {round((freq_dict[key] * 100), 2)}%')
        stats.write(f"{' '.join(string_list)}")
    else:
        weighted_var = pd.Series(weighted_num(selected_df, selected_df[variable_names[x]]))
        stats.write(f"**Mean**: {round(weighted_var.mean(), 2)}, **StDv**: {round(weighted_var.std(), 2)}, **Min**: {round(weighted_var.min(), 2)}, **1stQ**: {round(weighted_var.quantile(0.25), 2)}, **Median**: {round(weighted_var.median(), 2)}, **3stQ**: {round(weighted_var.quantile(0.75), 2)}, **Max**: {round(weighted_var.max(), 2)}")

# Hypothesis test
if one_variable_button == False:

    if str(selected_df[variable_names[x]].dtype) == 'object':

        ab_testing = st.beta_expander("A/B testing")

        weighted_df_x = pd.DataFrame(weighted_cat(selected_df, selected_df[variable_names[x]]))

        categories_x = sorted(selected_df[variable_names[x]].unique())
        categories_y = sorted(selected_df[variable_names[y]].unique())

        if str(selected_df[variable_names[y]].dtype) == 'object':

            weighted_df_y = pd.DataFrame(weighted_cat(selected_df, selected_df[variable_names[y]]))

            ohe = OneHotEncoder(categories=[categories_x])
            X = ohe.fit_transform(weighted_df_x).toarray()

            ohe = OneHotEncoder(categories=[categories_y])
            Y = ohe.fit_transform(weighted_df_y).toarray().transpose()

            for i in range(len(Y)):
                pvalues = chi2(X, Y[i])[1]
                string_list = []
                for i2 in range(len(categories_x)):
                    string_list.append(f'**{categories_x[i2]}**: {round(pvalues[i2], 3)}')
                ab_testing.write(f"***{categories_y[i]}*** p values --> {' '.join(string_list)}")

        else:
            weighted_df_y = pd.DataFrame(weighted_num(selected_df, selected_df[variable_names[y]]))

            scaler = MinMaxScaler()
            X = scaler.fit_transform(weighted_df_y)

            ohe = OneHotEncoder(categories=[categories_x])
            Y = ohe.fit_transform(weighted_df_x).toarray().transpose()

            string_list = []
            for i in range(len(Y)):
                pvalues = f_classif(X, Y[i])[1]
                string_list.append(f'**{categories_x[i]}**: {round(float(pvalues), 3)}')
            ab_testing.write(f"p values --> {' '.join(string_list)}")
    else:
        pass

# Subtitle for the prediction option
st.subheader('Check out the predictive model for multidimensional poverty!')

model_description = st.beta_expander("How is it done?")
model_description.write('''
This model is built with a Decision Tree algorithm. Decision trees are non-parametric supervised learning techniques used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
''')

# Setting variable inputs for prediction
with st.beta_container():
    col1, col2, col3, col4 = st.beta_columns(4)
    model_selected_sex = col1.selectbox('Sex', sex)
    model_selected_civil_status = col2.selectbox('Civil status', civil_status)
    model_selected_familial_status = col3.selectbox('Familial status', familial_status)
    model_selected_region = col4.selectbox('Region', region)
    model_selected_population_density = col1.selectbox('Population density', population_density)
    model_selected_citizenship = col2.selectbox('Citizenship', citizenship)
    model_selected_tenure_status = col3.selectbox('Tenure status', tenure_status)
    model_selected_education_level = col4.selectbox('Education level', education_level)
    model_selected_working_status = col1.selectbox('Working status', working_status)
    model_selected_occupation = col2.selectbox('Occupation', occupation)
    model_selected_bad_health = col3.selectbox('Bad health', bad_health)
    model_selected_age = col4.slider('Age', int(df.age.min()), int(df.age.max()))
    model_selected_years_worked = col1.slider('Years worked', int(df.years_worked.min()), int(df.years_worked.max()))
    model_selected_hours_week_worked = col2.slider('Hours a week worked', int(df.hours_week_worked.min()), int(df.hours_week_worked.max()))
    model_selected_adjusted_income = col3.slider('Adjusted income', -20000, 180000, -20000, 1000)
    model_selected_proportion_social_welfare = col4.slider('% social welfare', 0.0, 1.0, 0.0, 0.01)

#Loading data for model training
url2 = 'https://raw.githubusercontent.com/deividvalerius/Multidimensional-Poverty-Predictor/master/Data/to_model.csv'
to_model = pd.read_csv(url2)

X = to_model.drop(['material_deprivation'], axis=1)
y = to_model.material_deprivation

tree = DecisionTreeClassifier()
tree.fit(X, y)

#Input array for prediction
array = []

if model_selected_civil_status == 'Married':
    array.append(1)
else:
    array.append(0)

if model_selected_civil_status == 'Never married':
    array.append(1)
else:
    array.append(0)

if model_selected_civil_status == 'Separated':
    array.append(1)
else:
    array.append(0)

if model_selected_region == 'Basque Country':
    array.append(1)
else:
    array.append(0)

if model_selected_region == 'Castile–La Mancha':
    array.append(1)
else:
    array.append(0)

if model_selected_region == 'Andalusia':
    array.append(1)
else:
    array.append(0)

if model_selected_region == 'Castile and Leon':
    array.append(1)
else:
    array.append(0)

if model_selected_region == 'Cantabria':
    array.append(1)
else:
    array.append(0)

if model_selected_population_density == 'Thinly-populated area':
    array.append(1)
else:
    array.append(0)

if model_selected_citizenship == 'Spain':
    array.append(1)
else:
    array.append(0)

if model_selected_citizenship == 'Spain (naturalized)':
    array.append(1)
else:
    array.append(0)

if model_selected_citizenship == 'Other (outside EU)':
    array.append(1)
else:
    array.append(0)

if model_selected_citizenship == 'Other (EU)':
    array.append(1)
else:
    array.append(0)

if model_selected_tenure_status == 'Outright owner':
    array.append(1)
else:
    array.append(0)

if model_selected_tenure_status == 'Tenancy at reduced rate':
    array.append(1)
else:
    array.append(0)

if model_selected_tenure_status == 'Owner paying mortgage':
    array.append(1)
else:
    array.append(0)

if model_selected_tenure_status == 'Tenancy at market rate':
    array.append(1)
else:
    array.append(0)

if model_selected_tenure_status == 'Free tenancy':
    array.append(1)
else:
    array.append(0)

if model_selected_education_level == 'Lower secundary education':
    array.append(1)
else:
    array.append(0)

if model_selected_education_level == 'Higher education':
    array.append(1)
else:
    array.append(0)

if model_selected_education_level == 'Primary education':
    array.append(1)
else:
    array.append(0)

if model_selected_education_level == 'Pre-primary education':
    array.append(1)
else:
    array.append(0)

if model_selected_working_status == 'Retired':
    array.append(1)
else:
    array.append(0)

if model_selected_working_status == 'Disabled/unfit to work':
    array.append(1)
else:
    array.append(0)

if model_selected_working_status == 'Unemployed':
    array.append(1)
else:
    array.append(0)

if model_selected_working_status == 'Employed':
    array.append(1)
else:
    array.append(0)

if model_selected_occupation == 'Clerical Support Workers':
    array.append(1)
else:
    array.append(0)

if model_selected_occupation == 'Elementary Occupations':
    array.append(1)
else:
    array.append(0)

if model_selected_occupation == 'Professionals':
    array.append(1)
else:
    array.append(0)

if model_selected_occupation == 'Managers':
    array.append(1)
else:
    array.append(0)

if model_selected_occupation == 'Technicians and Associate Professionals':
    array.append(1)
else:
    array.append(0)

if model_selected_occupation == 'Non-defined':
    array.append(1)
else:
    array.append(0)

if model_selected_bad_health == 'Yes':
    array.append(1)
else:
    array.append(0)

if model_selected_bad_health == 'No':
    array.append(1)
else:
    array.append(0)

array.append((model_selected_age - min(df.age)) / (max(df.age) - min(df.age)))
array.append((model_selected_years_worked - min(df.years_worked)) / (max(df.years_worked) - min(df.years_worked)))
array.append((model_selected_hours_week_worked - min(df.hours_week_worked)) / (max(df.hours_week_worked) - min(df.hours_week_worked)))
array.append((model_selected_adjusted_income - min(df.adjusted_income)) / (max(df.adjusted_income) - min(df.adjusted_income)))
array.append((model_selected_proportion_social_welfare - min(df.proportion_social_welfare)) / (max(df.proportion_social_welfare) - min(df.proportion_social_welfare)))

#Prediction result
to_predict = np.array([array])
prediction = tree.predict_proba(to_predict)[0][1]

with st.beta_container():
    col1, col2, col3 = st.beta_columns([1.5, 3, 1])
    if prediction == 0.0:
        col2.markdown('### Not experiencing severe material deprivation')
    else:
        col2.markdown('### This person might be experiencing severe material deprivation!')
