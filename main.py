# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# ML Algorithm
import xgboost as xgb
# Success Metrics
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

class MLModel:
    FileFormats = ['xml','json','csv','excel']

    def __init__(self) -> None:
        self.dataFrame = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self, path: str):
        '''
            Yolu verilen dosyaya gore dataframe'i gunceller
        '''
        # Yolu verilen dosyanin formatinin belirlenmesi
        fileFormat = path.split('.')[-1]
        # Gecersiz dosya formatinda hata mesaji dondurulmesi
        if fileFormat.lower() not in MLModel.FileFormats: raise Exception("Unexpected file format!")
        # DataFramein initialize edilip edilmediginin kontrol edilmesi
        if not hasattr(self, "dataFrame"): raise Exception("Something wrong with the object initializing")
        # Algilanan dosya formatina gore DataFrame degiskeninin doldurulmasi
        command = "self.dataFrame = pd.read_{}('{}')".format(fileFormat, path)
        exec(command)
        
    def print_summary(self) -> None:
        '''
            Dataframe'in barindirdigi sutunlari analiz ederek 
            count, mean, std, min, 25%, 50%, 75%, max 
            degerlerini iceren matrix'i yazdiran fonksiyon
        '''
        print('NaN Values')
        print(self.dataFrame.isna().sum())
        print('Dataframe Summary')
        print(self.dataFrame.describe())
        print('Column Data Types')
        print(self.dataFrame.dtypes)
    

    def drop_extreme_vals(self):

        Q1 = self.dataFrame['Price'].quantile(0.25)
        Q3 = self.dataFrame['Price'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.15 * IQR
        upper_bound = Q3 + 0.75 * IQR

        target_indexes = self.dataFrame[(self.dataFrame['Price'] < lower_bound) | ((self.dataFrame['Price'] > upper_bound))].index
        self.dataFrame.drop(target_indexes, inplace=True)

        self.dataFrame = self.dataFrame.reset_index(drop=True)
        
    def draw_box_plot(self, x, y):
        sns.set(style="whitegrid")
        sns.boxplot(x=x, y=y, data=self.dataFrame)
        plt.show()

    def fit_transform_xgboost(self):
        model = xgb.XGBRegressor(n_estimators = 100, max_depth=7, objective = 'reg:squarederror', learning_rate = 0.5)
        model.fit(self.x_train, self.y_train)

        y_pred = model.predict(self.x_test)
        y_test = self.y_test.to_numpy()
        
        for i in range(len(y_test)):
            print('real:', y_test[i], 'predicted:', y_pred[i])
            
        r2 = metrics.r2_score(y_test, y_pred)
        print('r2:', r2)

    def drop_cols(self, columns):
        self.dataFrame.drop(columns=columns, inplace=True)

    def standardize_data(self, train):
        sc = StandardScaler()
        train = sc.fit_transform(train)
        return sc, train

    @staticmethod
    def draw_corr_matrix(matrix):
        sns.set(style="white")
        mask = np.triu(matrix, k=1)
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap='coolwarm', mask=mask)

        plt.title("Korelasyon Matrisi")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.show()

    @staticmethod
    def prepare_data(df, instance, split = False):
        rating_sorted = {'5 stars': 5, '4 stars': 4, '3 stars': 3, '2 stars': 2, '1 stars': 1}
        df['rating'] = df['rating'].apply(lambda cell: rating_sorted[cell])
        
        one_hot_encoder = OneHotEncoder()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            encoded_column = one_hot_encoder.fit_transform(df[[col]]).toarray()
            
            new_cols = pd.DataFrame(encoded_column, columns=one_hot_encoder.get_feature_names_out([col]))
            df = pd.concat([df, new_cols], axis=1)
            df.drop([col], axis=1, inplace=True)

        y = df['Price']
        x = df.drop('Price', axis=1)

        if split: instance.x_train, instance.x_test, instance.y_train, instance.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        return df

def main():
    mlModel = MLModel()
    mlModel.load_data('./data/laptopPrice.csv')
    mlModel.drop_cols({'Touchscreen', 'msoffice', 'Number of Reviews'})
    mlModel.drop_extreme_vals()
    mlModel.dataFrame = MLModel.prepare_data(mlModel.dataFrame, mlModel, True)
    mlModel.fit_transform_xgboost()
    


if __name__ == '__main__':
    main()