from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss,confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
import mleap.sklearn.preprocessing.data
from mleap.sklearn.preprocessing.data import FeatureExtractor
import mleap.sklearn.pipeline
import pickle
import pandas as pd

marketing=pd.read_csv("data/marketing.csv",index_col=0).dropna()


X = marketing[['marketing_channel','subscribing_channel','age_group']]
y = marketing['converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Feature Selector: select categorical columns
input_features = X_train.select_dtypes(include=['object']).columns.to_list()
output_vector_name = 'unscaled_continuous_features' # for serialization
output_features = ["{}_encoded".format(x) for x in input_features]

# initialize feature extractor transformer for serialization
feature_extractor_tf = FeatureExtractor(input_scalars=input_features,
                                        output_vector=output_vector_name,
                                        output_vector_items=output_features)

#initialize imputer to fill in NAs
# imp=SimpleImputer(missing_values=np.nan, fill_value='missing')                                     
# imp.mlinit(prior_tf=feature_extractor_tf, output_features='imp_features')

# initialize one hot encoder & pass prior transformer
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.mlinit(prior_tf=imp, output_features='encoded_features')

pipe = Pipeline(steps=[
    ('feat_extract',feature_extractor_tf),
    ('onehot',ohe)])

pipe.fit_transform(X_train)
pipe.mlinit()

pipe.serialize_to_bundle('/tmp','mleap-bundle', init=True)