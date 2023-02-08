import pandas as pd
import numpy as np
from tqdm import tqdm
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

def readFile():
    '''
    read and merge the two excel files
    '''
    sales_fact = pd.read_csv(r"C:\Users\USER\Desktop\data mining\hw1_data\sales_fact_1998.csv")
    sales_dec = pd.read_csv(r"C:\Users\USER\Desktop\data mining\hw1_data\sales_fact_dec_1998.csv")
    sales = pd.concat([sales_fact, sales_dec])
    return sales

def buildFrequentItemsets(sales):
    '''
    Build frequent itemSets using the merged pandas dataFrames
    find each transaction data based on the same purchase time, store, and customer_id
    each transaction is a list, so we return a list of transaction lists
    '''
    #sales merge product: #sales --- product id 對應 product的product class --- 對應prodoct_class 的product_department
    product = pd.read_csv(r"C:\Users\USER\Desktop\data mining\hw1_data\product.csv")
    product_class = pd.read_csv(r"C:\Users\USER\Desktop\data mining\hw1_data\product_class.csv")
    product_csv = (pd.merge(product, product_class, on='product_class_id'))[['product_id', 'product_department']]
    #print(product_csv)
    # merge_csv = pd.merge(product_csv, merge_csv, on='product_id')
    merge_csv = pd.merge(product_csv, sales, on='product_id')
    productdf = merge_csv.groupby(['time_id', 'store_id', 'customer_id'])['product_department'].apply(list)
    unitdf = merge_csv.groupby(['time_id', 'store_id', 'customer_id'])['unit_sales'].apply(list)
    # print(productdf)
    # print(unitdf)
    
    transactionList = list()
    for productList, unitList in zip(productdf, unitdf):#multiple transactions
        transProduct = [product for product, unit in zip(productList, unitList) for i in range(int(unit))]
        transactionList.append(transProduct)
    #print(len(transactionList))
    return transactionList

def associationApriori(itemSets, continueMode):
    '''
    given a frequent itemSets(a list of lists)
    output the sorted association results by confidence and by lift respectively
    if continueMode == 1, we will print the result directly
    '''
    encoder = TransactionEncoder()
    transBool = encoder.fit(itemSets).transform(itemSets)# a list of lists of boolean
    #print(encoder.columns_)#print columns of arr---products
    df = pd.DataFrame(transBool, columns=encoder.columns_)#arr to dataframe
    
    if continueMode == 0:
        support = apriori(df, min_support=0.00005, use_colnames=True, low_memory=True)
        associationRules = (association_rules(support, metric='confidence', min_threshold=0.9, support_only=False)).sort_values('support', ascending=False)
        byConfidence = associationRules.sort_values('confidence', ascending=False)
        byLift = associationRules.sort_values('lift', ascending=False)
        return byConfidence[['antecedents', 'consequents', 'support', 'confidence', 'lift']], byLift[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    elif continueMode == 1:
        support = apriori(df, min_support=0.0001, use_colnames=True, low_memory=True)###
        associationRules = (association_rules(support, metric='confidence', min_threshold=0.9, support_only=False)).sort_values('support', ascending=False)
        byConfidence = associationRules.sort_values('confidence', ascending=False)
        byLift = associationRules.sort_values('lift', ascending=False)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):###
            # print(byConfidence.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
            print(byLift.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
            print('------------------------------------------------------------------------')
            df = byLift.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            printdf(df)
    return byConfidence.head(10), byLift.head(10)

def printdf(df):
    lst = [sorted(list(fs)) for fs in df['antecedents']]
    lst_c = [sorted(list(fs)) for fs in df['consequents']]
    for c, (i, j) in enumerate(zip(lst, lst_c)):
        print(f'rule {c+1}', end='\t')
        print(f'{i} -> {j}')
        
def fp_growth(itemSets):
    '''
    given a frequent itemSets(a list of lists)
    output FP Growth result sorted by confidence and lift
    '''
    #print(itemSets)
    encoder = TransactionEncoder()
    transBool = encoder.fit(itemSets).transform(itemSets)# a list of lists of boolean
    #print(encoder.columns_)#print columns of arr---products
    df = pd.DataFrame(transBool, columns=encoder.columns_)#arr to dataframe
    support = fpgrowth(df, min_support=0.0001, use_colnames=True)
    associationRules = (association_rules(support, metric='confidence', min_threshold=0.9, support_only=False)).sort_values('support', ascending=False)
    #print(associationRules)
    byConfidence = associationRules.sort_values('confidence', ascending=False)
    byLift = associationRules.sort_values('lift', ascending=False)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # print(byConfidence.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        print(byLift.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    print('---------------------------------------------------------------------------------------------------------')

    df = byLift.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    printdf(df)
    
    return byConfidence.head(10), byLift.head(10)

def readUser():
    '''
    read user profile data csv
    output a list of lists
    each list is a customer profile row in csv
    '''
    user_csv = pd.read_csv(r"C:\Users\USER\Desktop\data mining\hw1_data\customer.csv")
    user_csv = user_csv.applymap(str)
    user_csv['total_children'] = user_csv['total_children'].apply(lambda x: f"{x} totalChildren")
    user_csv['num_children_at_home'] = user_csv['num_children_at_home'].apply(lambda x: f"{x} homeChildren")
    user_csv['num_cars_owned'] = user_csv['num_cars_owned'].apply(lambda x: f"{x} carNums")
    user_list = user_csv[['state_province', 'yearly_income', 'gender', 'total_children',
                    'num_children_at_home', 'education', 'occupation', 'houseowner',
                    'num_cars_owned']].values.tolist()
    #print(user_list[:5])
    return user_list
def merge_csvs(sales):
    '''
    merge three csv(in fact there are 4): customer and sales(already merged in readFile() function)
        sales vs. customer: merge according to customer_id(like vlookup) o.w. the two csv have different number of rows
        product vs. merger: merge according to product_id
    combine customer data with sales data
    output a list of lists, each list specifies customer info, purchase items and number of items of each customer
    '''
    user_csv = pd.read_csv(r"C:\Users\USER\Desktop\data mining\hw1_data\customer.csv")
    birth_info_list = user_csv['birthdate'].str.split('/')
    lst = list()
    for cell in birth_info_list:
        if 0 <= (1998 - int(cell[0])) <= 29:
            cell[0] = 'young'
        elif 30 <= (1998 - int(cell[0])) <= 59:
            cell[0] = 'middle'
        elif (1998 - int(cell[0])) >= 60:
            cell[0] = 'old'
        lst.append(cell[0])
    user_csv['age'] = pd.Series(lst)
    user_csv['birth_month'] = pd.Series([str(cell[1])  + ' month' for cell in birth_info_list])
    user_csv = user_csv[['customer_id', 'postal_code', 'yearly_income', 'marital_status', 'gender', 'total_children',
                         'education', 'member_card', 'occupation', 'houseowner',
                         'num_cars_owned', 'age', 'birth_month']]
    #print(user_csv.info())
    
    #sales merge user: customer_id
    sales_csv = sales.drop(['store_sales', 'store_cost'], axis=1)
    merge_csv = pd.merge(user_csv, sales_csv, on='customer_id')
    #print(merge_csv)
    
    #sales merge product: #sales --- product id 對應 product的product class --- 對應prodoct_class 的product_department
    product = pd.read_csv(r"C:\Users\USER\Desktop\data mining\hw1_data\product.csv")
    product_class = pd.read_csv(r"C:\Users\USER\Desktop\data mining\hw1_data\product_class.csv")
    product_csv = (pd.merge(product, product_class, on='product_class_id'))[['product_id', 'product_department']]
    #print(product_csv)
    merge_csv = pd.merge(product_csv, merge_csv, on='product_id')
    
    #preprocessing
    #merge_csv = merge_csv.applymap(str)
    merge_csv['total_children'] = merge_csv['total_children'].apply(lambda x: f"{x} totalChildren")
    merge_csv['num_cars_owned'] = merge_csv['num_cars_owned'].apply(lambda x: f"{x} carNums")
    merge_csv['postal_code'] = merge_csv['postal_code'].apply(lambda x: f"{x} postalCode")
    merge_csv['marital_status'] = merge_csv['marital_status'].str.replace('M', 'married')
    #print(merge_csv)
    
    return merge_csv

def merge_trans_user(sales, based):
    merge_csv = merge_csvs(sales)
    #GROUP BY
    groupbyer = merge_csv.groupby(['time_id', 'store_id', 'customer_id'])
    productdf = groupbyer['product_department'].apply(list)
    unitdf = groupbyer['unit_sales'].apply(list)
    
    #build a list of transaction lists
    transactionList = list()
    for productList, unitList in zip(productdf, unitdf):#multiple transactions
        transProduct = [product for product, unit in zip(productList, unitList) for i in range(int(unit))]
        #print(transProduct)
        transactionList.append(transProduct)
    # for col in ['postal_code', 'yearly_income', 'marital_status', 'gender', 'total_children','education', 'member_card', 
    #             'occupation', 'houseowner', 'num_cars_owned', 'age', 'birth_month']:
    col = based
    for person_info, transaction in zip(groupbyer[col].apply(list), transactionList):
        transaction.append(person_info[0])# [0] since person_info is unique but is repeated due to group by
        #print(transaction)
    based_valueList = merge_csv[based].unique()
    print(based_valueList)
    
    for based in based_valueList:
        byCon, byLift = associationApriori(transactionList, 0)
        #print(byCon[byCon['consequents'] == frozenset({'F'})])
        #print(len(byLift[byLift['consequents'] == frozenset({based})]))
        #print('----------------------------------------------------------------------------')
        #print(len(byLift))
        byLift = byLift[byLift['consequents'] == frozenset({based})]
        temp = byLift['antecedents']
        print(temp)
        print('----------------------------------------------------------------------------')
        #print(temp.unique())
    return merge_csv

def novCompare(sales):
    #要用sales data中的time_id查表
    sales = merge_csvs(sales)
    date_csv = pd.read_csv(r"C:\Users\USER\Desktop\data mining\hw1_data\time_by_day.csv")
    merge_csv= pd.merge(date_csv, sales, on='time_id')
    #print(merge_csv.columns)
    
    #change value of the_year GROUP BY using the_year
    lst = list()
    for cell in merge_csv['the_month']:
        if cell == 'December':
            cell = 'Dec'
        else:
            cell = 'notDec'
        lst.append(cell)
    merge_csv['the_month'] = pd.Series(lst)
    merge_csv = merge_csv[['the_month', 'product_department', 'customer_id', 'time_id', 'store_id', 'promotion_id', 'unit_sales']]
    #be careful promotion
    dec_csv = merge_csv[merge_csv['the_month'] == 'Dec']
    not_dec_csv = merge_csv[merge_csv['the_month'] == 'notDec']
    
    #GROUP BY
    for csv in dec_csv, not_dec_csv:
        groupbyer = csv.groupby(['time_id', 'store_id', 'customer_id'])
        productdf = groupbyer['product_department'].apply(list)
        unitdf = groupbyer['unit_sales'].apply(list)
        
        #build a list of transaction lists
        transactionList = list()
        for productList, unitList in zip(productdf, unitdf):#multiple transactions
            transProduct = [product for product, unit in zip(productList, unitList) for i in range(int(unit))]
            #print(transProduct)
            transactionList.append(transProduct)
        byC, byL = associationApriori(transactionList, 0)
        byL['len'] = byL['antecedents'].apply(lambda x: len(x)) + byL['consequents'].apply(lambda x: len(x))
        print(byL.head(10))
        print('-----------------------------------------')
    
    
    return merge_csv

if __name__ == '__main__':
    pd.set_option('display.max_colwidth', None)
    sales = readFile()
    '''
    第一題(Apriori)、第二題(FP)
    '''
    # itemSets = buildFrequentItemsets(sales)#each item is a int so far
    # # apConfidence, apLift = associationApriori(itemSets, 1)
    # fpConfidence, fpLift = fp_growth(itemSets)
    '''
    第三題
    # '''
    # user_df = readUser()
    # userApConfidence, userApLift = associationApriori(user_df, 1)
    # fpConfidence, fpLift = fp_growth(user_df)
    
    '''
    第四題
    '''
    based = ['postal_code', 'yearly_income', 'marital_status', 'gender', 'total_children','education', 'member_card', 
     'occupation', 'houseowner', 'num_cars_owned', 'age', 'birth_month']#有無小孩
    transactionList = merge_trans_user(sales, 'total_children')###
    # apply association rule
    apConfidence, apLift = associationApriori(transactionList, 0)
    # fpConfidence, fpLift = fp_growth(transactionList)
    
    #visualization? how to summarize
    
    '''
    第五題
    '''
    #novCompare(sales)