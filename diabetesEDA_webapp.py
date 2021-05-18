import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from PIL import Image
# 

def main():
    st.write(""" 
    # Diabetes Detection
    Detect if someone has diabetes using machine learning and python !
    """)

    image = Image.open('C:/Users/anike/Desktop/colg-Final/diabetes_prediction_project/Final Year Project Final/diabetes_logo.png')
    st.image(image, caption='ML', use_column_width=True)

    html_temp = """ 
            <div style="background-color:tomato;"><p style="color:white; font-size:50px;">Exploratory Data Analysis</p></div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)

    def file_selector(folder_path='./dataset'):
            filenames = os.listdir(folder_path)
            selected_filename = st.selectbox("Select a file", filenames)
            return os.path.join(folder_path, selected_filename)
    filename = file_selector()
    st.info("You Selected {}".format(filename))

    #Read Data
    df = pd.read_csv(filename)
    df.drop('Unnamed: 0',axis=1,inplace=True)
    df.drop('Unnamed: 0.1',axis=1,inplace=True)
    # Show dataset
    if st.checkbox("Show Dataset"):
        number = st.number_input("Number of Rows to View", 1, 100)
        st.dataframe(df.head(number))
    # Show Columns
    if st.button("Column Names"):
        st.write(df.columns)

    #Show Shape
    if st.checkbox("Shape of DataSet"):
        data_dim = st.radio("Show Dimension By ", ("Rows", "Columns"))
        if data_dim == 'Rows':
            st.text("Number of Columns")
            st.write(df.shape[0])
        if data_dim == 'Columns':
            st.text("Number of Columns")
            st.write(df.shape[1])
        else:
            st.write(df.shape)

    # Select Columns
    if st.checkbox("Select Columns to Show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select", all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    # Show Values
    if st.button("Value Counts"):
        st.text("Value Counts by Target/Class")
        st.write(df.iloc[: , -1].value_counts())

    # Show DataTypes
    if st.button("Data Types"):
        # st.text("Data Types")
        st.write(df.dtypes)

    # Show Summary
    if st.checkbox("Summary"):
        st.write(df.describe().T)

    
    # Distributions Plots
    st.write(" # Visualizations")
    # Pie Chart for % of Diabetes and Healthy patients
    st.markdown("**Pie Chart for Healthy and Diabetice**")
    if st.checkbox("Pie Plot",):
        sns.set(style="whitegrid")
        labels = ['Healthy', 'Diabetic']
        sizes = df['Outcome'].value_counts(sort = True)
        colors = ["lightblue","red"]
        explode = (0.05,0) 
        plt.figure(figsize=(5,5))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90,)
        plt.title('Number of diabetes in the dataset')
        plt.show()
        st.pyplot()
        st.markdown("**From abve plot we can say that around 65% are healthy and 35% are diabetic.**")

    # Distributin Plots
    # Pregnancy Dist Plot
    st.markdown("**Healthy vs Diabetic by Pregnancies**")
    if st.checkbox("Dist for Pregancies"):
        # Creating 3 subplots 
        # Plotting for pregnancies
        #Histogram 1
        plt.figure(figsize=(20, 6))
        plt.subplot(1,3,1)
        sns.set_style("dark")
        plt.title("Histogram for Pregnancies")
        sns.distplot(df.Pregnancies,kde=False)
        #Histogram 2
        plt.subplot(1,3,2)
        sns.distplot(df[df['Outcome'] == 0]["Pregnancies"], color='green') # Healthy - green
        sns.distplot(df[df['Outcome'] == 1]["Pregnancies"], color='red') # Diabetic - Red
        plt.title("Histograms for Preg by Outcome")
        plt.legend()
        #BoxPlot
        plt.subplot(1,3,3)
        sns.boxplot(x=df.Outcome,y=df.Pregnancies)
        plt.title("Boxplot for Preg by Outcome")
        plt.show()
        st.pyplot()
        st.markdown("**From above graph, we can say that the Pregnancy isn't likely cause for diabetes as the distribution between the Healthy and Diabetic is almost same**")
        st.markdown("**Visually, data is right skewed. For data of count of pregenancies. A large proportion of the participants are zero count on pregnancy. As the data set includes women > 21 yrs, its likely that many are unmarried \
                    When looking at the segemented histograms, pregnent women with 6-9 pregnancies were moslty diagonized as diabetic**")

    # Healthy vs Diabetic by Glucose
    st.markdown("**Healthy vs Diabetic by Glucose**")
    if st.checkbox("Dist for Glucose"):
        #Plotting Glucose Variable
        # Histogram 1
        plt.figure(figsize=(20, 6))
        plt.subplot(1,3,1)
        plt.title("Histogram for Glucose")
        sns.distplot(df.Glucose, kde=False)
        #Histogram 2
        plt.subplot(1,3,2)
        sns.distplot(df[df['Outcome'] == 0]["Glucose"],kde=False, color='green') # Healthy - green
        sns.distplot(df[df['Outcome'] == 1]["Glucose"],kde=False, color='red') # Diabetic - Red
        plt.title("Histograms for Glucose by Outcome")
        plt.legend()
        #BoxPlot
        plt.subplot(1,3,3)
        sns.boxplot(x=df.Outcome,y=df.Glucose)
        plt.title("Boxplot for Glucose by Outcome")
        plt.show()
        st.pyplot()
        st.markdown("**Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. \
                    The Glucose level for a Normal Adult is around 120-130mg/dl anything above it means that the person is likely suffering from pre-diabetes and diabetes. \
                    From above graph, we can see the the Healthy person are more around 120mg/dl but it then gradually drops, and for diabetic person it is vice versa.**")

    # Healthy vs Diabetic for BloodPressure
    st.markdown("**Healthy vs Diabetic by BloodPressure**")
    if st.checkbox("Dist for BloodPressure"):
        # plotting for Bloodpressure
        # Histogram 1 
        plt.figure(figsize=(20, 6))
        plt.subplot(1,3,1)
        sns.distplot(df.BloodPressure, kde=False)
        plt.title("Histogram for Blood Pressure")
        #Histogram 2
        plt.subplot(1,3,2)
        sns.distplot(df[df['Outcome'] == 0]["BloodPressure"],kde=False,color="green",label="BP for Outcome=0")
        sns.distplot(df[df['Outcome'] == 1]["BloodPressure"],kde=False, color="red", label="BP for Outcome=1")
        plt.legend()
        plt.title("Histogram of Blood Pressure by Outcome")
        #BoxPlot
        plt.subplot(1,3,3)
        sns.boxplot(x=df.Outcome,y=df.BloodPressure)
        plt.title("Boxplot of BP by Outcome")
        plt.show()
        st.pyplot()
        st.markdown("**High blood pressure (also known as “hypertension”) is very common in people with diabetes. In fact, the two conditions often go hand-in-hand because they can both result from the same lifestyle factors. \
                    Diabetes damages arteries and makes them targets for hardening, called atherosclerosis. That can cause high blood pressure, which if not treated, can lead to trouble including blood vessel damage, heart attack, and kidney failure. \
                    For a Normal person BP should be at or below 120/80 mm Hg, the person with hypertension can be above 139/89 mm Hg. \
                    From above graph, we can say that, diabetic and healthy people are evenly distributed with low and normal BP but, there are less healthy people who have high BP.**")
    
    # Healthy vs Diabetic for SkinThickness
    st.markdown("**Healthy vs Diabetic by SkinThickness**")
    if st.checkbox("Dist for SkinThickness"):
        # plotting for SkinThickness
        # Histogram 1 
        plt.figure(figsize=(20, 6))
        plt.subplot(1,3,1)
        sns.distplot(df.SkinThickness, kde=False)
        plt.title("Histogram for Skin Thickness")
        #Histogram 2
        plt.subplot(1,3,2)
        sns.distplot(df[df['Outcome'] == 0]["SkinThickness"],kde=False,color="green",label="SkinThickness for Outcome=0")
        sns.distplot(df[df['Outcome'] == 1]["SkinThickness"],kde=False, color="red", label="SkinThickness for Outcome=1")
        plt.legend()
        plt.title("Histogram for SkinThickness by Outcome")
        #BoxPlot
        plt.subplot(1,3,3)
        sns.boxplot(x=df.Outcome, y=df.SkinThickness)
        plt.title("Boxplot of SkinThickness by Outcome")
        plt.show()
        st.pyplot()
        st.markdown("**Changes to the blood vessels because of diabetes can cause a skin condition called diabetic dermopathy. Dermopathy appears as scaly patches that are light brown or red, often on the front of the legs. The patches do not hurt, blister, or itch, and treatment generally is not necessary. \
                    From above graph, the distribution between healthy and diabetic people are around same for skin thickness.**")

    # Healthy vs Diabetic for Insulin
    st.markdown("**Healthy vs Diabetic by Insulin**")
    if st.checkbox("Dist for Insulin"):
        # plotting for Insulin
        # Histogram 1 
        plt.figure(figsize=(20, 6))
        plt.subplot(1,3,1)
        sns.distplot(df.Insulin,kde=False)
        plt.title("Histogram of Insulin")
        # Histogram 2 
        plt.subplot(1,3,2)
        sns.distplot(df[df['Outcome'] == 0]["Insulin"],kde=False,color="green",label="Insulin for Outcome=0")
        sns.distplot(df[df['Outcome'] == 1]["Insulin"],kde=False, color="red", label="Insulin for Outcome=1")
        plt.title("Histogram for Insulin by Outcome")
        plt.legend()
        # Boxplot
        plt.subplot(1,3,3)
        sns.boxplot(x=df.Outcome, y=df.Insulin)
        plt.title("Boxplot for Insulin by Outcome")
        plt.show()
        st.pyplot()
        st.markdown("**Insulin is a hormone that your pancreas makes to allow cells to use glucose. When your body isn't making or using insulin correctly, you can take man-made insulin to help control your blood sugar. Many types can be used to treat diabetes. \
                    Insulin helps control blood glucose levels by signaling the liver and muscle and fat cells to take in glucose from the blood. Insulin therefore helps cells to take in glucose to be used for energy. If the body has sufficient energy, insulin signals the liver to take up glucose and store it as glycogen. \
                    From above graph, we can see that there are diabetic people increase as the levels of insulin gradually increases. There are more healthy people around insulin levels 0-100.**")
    
    # Healthy vs Diabetic for BMI
    st.markdown("**Healthy vs Diabetic by BMI**")
    if st.checkbox("Dist for BMI"):
        # plotting for BMI
        # Histogram 1 
        plt.figure(figsize=(20, 6))
        plt.subplot(1,3,1)
        sns.distplot(df.BMI, kde=False)
        plt.title("Histogram for BMI")
        # Histogram 2
        plt.subplot(1,3,2)
        sns.distplot(df[df['Outcome'] == 0]["BMI"],kde=False,color="green",label="BMI for Outcome=0")
        sns.distplot(df[df['Outcome'] == 1]["BMI"],kde=False, color="red", label="BMI for Outcome=1")
        plt.legend()
        plt.title("Histogram for BMI by Outcome")
        # BoxPlot
        plt.subplot(1,3,3)
        sns.boxplot(x=df.Outcome, y=df.BMI)
        plt.title("Boxplot for BMI by Outcome")
        plt.show()
        st.pyplot()
        st.markdown("**Being overweight (BMI of 25-29.9), or affected by obesity (BMI of 30-39.9) or morbid obesity (BMI of 40 or greater), greatly increases your risk of developing type 2 diabetes. The more excess weight you have, the more resistant your muscle and tissue cells become to your own insulin hormone. \
                    From above graph we can determine that, as the BMI increases the person likely being healthy decreases and being diabetic increases.**")

    # Healthy vs Diabetic for DPF
    st.markdown("**Healthy vs Diabetic by Diabetes Pedigree Function**")
    if st.checkbox("Dist for DPF"):
        # plotting for DPF
        # Histogram 1 
        plt.figure(figsize=(20, 6))
        plt.subplot(1,3,1)
        sns.distplot(df.DiabetesPedigreeFunction, kde=False)
        plt.title("Histogram for BMI")
        # Histogram 2
        plt.subplot(1,3,2)
        sns.distplot(df[df['Outcome'] == 0]["DiabetesPedigreeFunction"],kde=False,color="green",label="DPF for Outcome=0")
        sns.distplot(df[df['Outcome'] == 1]["DiabetesPedigreeFunction"],kde=False, color="red", label="DPF for Outcome=1")
        plt.legend()
        plt.title("Histogram for BMI by Outcome")
        # BoxPlot
        plt.subplot(1,3,3)
        sns.boxplot(x=df.Outcome, y=df.DiabetesPedigreeFunction)
        plt.title("Boxplot for BMI by Outcome")
        plt.show()
        st.pyplot()
        st.markdown("**Diabetes Pedigree Function is a function which scores likelihood of diabetes based on family history. It provided some data on diabetes mellitus history in relatives and the genetic relationship of those relatives to the patient. \
                    From above graph, as thefunction increase the diabetic people increases, showing that the diabetes could be hereditary for that individual.**")

    # Healthy vs Diabetic for Age
    st.markdown("**Healthy vs Diabetic by Age**")
    if st.checkbox("Dist for Age"):
        # plotting for Age
        # Histogram 1 
        plt.figure(figsize=(20, 6))
        plt.subplot(1,3,1)
        sns.distplot(df.Age, kde=False)
        plt.title("Histogram for BMI")
        # Histogram 2
        plt.subplot(1,3,2)
        sns.distplot(df[df['Outcome'] == 0]["Age"],kde=False,color="green",label="Age for Outcome=0")
        sns.distplot(df[df['Outcome'] == 1]["Age"],kde=False, color="red", label="Age for Outcome=1")
        plt.legend()
        plt.title("Histogram for BMI by Outcome")
        # BoxPlot
        plt.subplot(1,3,3)
        sns.boxplot(x=df.Outcome, y=df.Age)
        plt.title("Boxplot for BMI by Outcome")
        plt.show()
        st.pyplot()
        st.markdown("**As the person ages, they are at high risk for the development of type 2 diabetes due to the combined effects of increasing insulin resistance and impaired pancreatic islet function with aging. \
                    From above graph, we can see that there are more healthy people around 20-25 age but as the age gradually increases so does the people being diabetic, this shows that age and diabetes go hand in hand.**")

    #Correaltion
    st.markdown("**Correlation Plot**")
    if st.checkbox("Correlation Plot"):
        plt.figure(figsize=(10,10))
        sns.heatmap(df.corr(method='pearson'), cmap="YlGnBu", annot= True,)
        plt.show()
        st.pyplot()
        st.markdown("**No 2 factors have strong linear relationships ,Age & Pregnancies and BMI & SkinThickness have moderate positive linear relationship. Glucose & Insulin technically has low correlation but 0.57 is close to 0.6 so can be assumed as moderate correlation.**")

    # Pairplot
    st.markdown("**PairPlot **")
    if st.checkbox("Pairplot"):
        sns.pairplot(df, vars=["Pregnancies", "Glucose","BloodPressure","SkinThickness","Insulin", "BMI","DiabetesPedigreeFunction", "Age"],hue="Outcome")
        plt.title("Pairplot of Variables by Outcome")
        plt.show()
        st.pyplot()

if __name__ == '__main__':
    main()