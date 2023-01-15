# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics

# Any results you write to the current directory are saved as output.

LAST_SUFFIX='_last'
MEAN_SUFFIX='_mean'
MEAN_2YRS_SUFFIX='_mean2'
#MEAN_3YRS_SUFFIX='_mean3'
competition_year = 2018
train_first_year = 2003

default_seed = 8
train_offset = 10
predict_year = 2018

train_year = predict_year - 1
IS_VALIDATION = predict_year != competition_year


#Add to season matchups in train historical data going back to yearStart
#W/L ratio
#respective team info median over 
#season = 2017
#yearStart = 2003
#current_season = train_current_season_tmp
#all_data = hist_data
def createFeatures(all_data, season, yearStart, current_season, df_ordinals_filtered):
    season_data = current_season[['TeamID1','TeamID2','Team1_Seed','Team2_Seed']]
    #greater equal doesn't compile???
    df_tmp = all_data.loc[(all_data['Season'] < season) & (all_data['Season'] > yearStart-1)]

    ####################################################### W/L RATIOS ####################################################### 
    #calc mean feature
    df_tmp = df_tmp.join(df_tmp.groupby(['TeamID1','TeamID2'])['Result'].mean(), on=['TeamID1','TeamID2'], rsuffix=MEAN_SUFFIX)
    #df_tmp = df_tmp.merge(df_tmp.groupby(['TeamID1','TeamID2'], as_index=False)['Result'].mean(), how='left', on=['TeamID1','TeamID2'], suffixes=[None,MEAN_SUFFIX])
    #get lastyear features
    df_tmp_lastyear = df_tmp.loc[df_tmp['Season'] == season-1]
    df_tmp = df_tmp.merge(df_tmp_lastyear[['TeamID1','TeamID2','Result']], how='left', on=['TeamID1','TeamID2'], suffixes=[None,LAST_SUFFIX])
    #fill NaN's as needed
    df_tmp['Result'+LAST_SUFFIX] = df_tmp['Result'+LAST_SUFFIX].fillna(df_tmp['Result'+MEAN_SUFFIX])
    #then, merge with current season data
    #but before, remove duplicates. They'll all have the same mean and lastyear data anyway
    df_tmp = df_tmp.drop_duplicates(subset=['TeamID1','TeamID2'])
    
    ##### MERGE
    feature_data = season_data.merge(df_tmp[['TeamID1','TeamID2','Result'+LAST_SUFFIX,'Result'+MEAN_SUFFIX]], how='left', on=['TeamID1','TeamID2'])
    #fill NaN's as needed
    #for now we naively assume 0.5
    feature_data['Result'+LAST_SUFFIX] = feature_data['Result'+LAST_SUFFIX].fillna(0.5)
    feature_data['Result'+MEAN_SUFFIX] = feature_data['Result'+MEAN_SUFFIX].fillna(0.5)

    ####################################################### GAME FEATURES ####################################################### 
    #Points Team1 / Team 2
    df_tmp['Team1_Pts'] = df_tmp.apply(lambda row: 2*row.Team1_FGM + row.Team1_FGM3 + row.Team1_FTM, axis=1)
    df_tmp['Team2_Pts'] = df_tmp.apply(lambda row: 2*row.Team2_FGM + row.Team2_FGM3 + row.Team2_FTM, axis=1)


    #Calculate Team Possesion Feature
    pos1 = df_tmp.apply(lambda row: 0.96*(row.Team1_FGA + row.Team1_TO + 0.44*row.Team1_FTA - row.Team1_OR), axis=1)
    pos2 = df_tmp.apply(lambda row: 0.96*(row.Team2_FGA + row.Team2_TO + 0.44*row.Team2_FTA - row.Team2_OR), axis=1)
    #two teams use almost the same number of possessions in a game
    #(plus/minus one or two - depending on how quarters end)
    #so let's just take the average
    df_tmp['Pos1'] = (pos1+pos2)/2
    df_tmp['Pos2'] = df_tmp['Pos1']
    #Offensive efficiency (OffRtg) = 100 x (Points / Possessions)
    df_tmp['Team1_OffRtg'] = df_tmp.apply(lambda row: 100 * (row.Team1_Pts / row.Pos1), axis=1)
    df_tmp['Team2_OffRtg'] = df_tmp.apply(lambda row: 100 * (row.Team2_Pts / row.Pos2), axis=1)
    #Defensive efficiency (DefRtg) = 100 x (Opponent points / Opponent possessions)
    df_tmp['Team1_DefRtg'] = df_tmp.Team2_OffRtg
    df_tmp['Team2_DefRtg'] = df_tmp.Team1_OffRtg
    #Net Rating = Off.Rtg - Def.Rtg
    df_tmp['Team1_NetRtg'] = df_tmp.apply(lambda row:(row.Team1_OffRtg - row.Team1_DefRtg), axis=1)
    df_tmp['Team2_NetRtg'] = df_tmp.apply(lambda row:(row.Team2_OffRtg - row.Team2_DefRtg), axis=1)
                         
    #Assist Ratio : Percentage of team possessions that end in assists
    df_tmp['Team1_AstR'] = df_tmp.apply(lambda row: 100 * row.Team1_Ast / (row.Team1_FGA + 0.44*row.Team1_FTA + row.Team1_Ast + row.Team1_TO), axis=1)
    df_tmp['Team2_AstR'] = df_tmp.apply(lambda row: 100 * row.Team2_Ast / (row.Team2_FGA + 0.44*row.Team2_FTA + row.Team2_Ast + row.Team2_TO), axis=1)
    #Turnover Ratio: Number of turnovers of a team per 100 possessions used.
    #(TO * 100) / (FGA + (FTA * 0.44) + AST + TO)
    df_tmp['Team1_TOR'] = df_tmp.apply(lambda row: 100 * row.Team1_TO / (row.Team1_FGA + 0.44*row.Team1_FTA + row.Team1_Ast + row.Team1_TO), axis=1)
    df_tmp['Team2_TOR'] = df_tmp.apply(lambda row: 100 * row.Team2_TO / (row.Team2_FGA + 0.44*row.Team2_FTA + row.Team2_Ast + row.Team2_TO), axis=1)
                        
    #The Shooting Percentage : Measure of Shooting Efficiency (FGA/FGA3, FTA)
    df_tmp['Team1_TSP'] = df_tmp.apply(lambda row: 100 * row.Team1_Pts / (2 * (row.Team1_FGA + 0.44 * row.Team1_FTA)), axis=1)
    df_tmp['Team2_TSP'] = df_tmp.apply(lambda row: 100 * row.Team2_Pts / (2 * (row.Team2_FGA + 0.44 * row.Team2_FTA)), axis=1)
    #eFG% : Effective Field Goal Percentage adjusting for the fact that 3pt shots are more valuable 
    df_tmp['Team1_eFGP'] = df_tmp.apply(lambda row:(row.Team1_FGM + 0.5 * row.Team1_FGM3) / row.Team1_FGA, axis=1)      
    df_tmp['Team2_eFGP'] = df_tmp.apply(lambda row:(row.Team2_FGM + 0.5 * row.Team2_FGM3) / row.Team2_FGA, axis=1)   
    #FTA Rate : How good a team is at drawing fouls.
    df_tmp['Team1_FTAR'] = df_tmp.apply(lambda row: row.Team1_FTA / row.Team1_FGA, axis=1)
    df_tmp['Team2_FTAR'] = df_tmp.apply(lambda row: row.Team2_FTA / row.Team2_FGA, axis=1)
                             
    #OREB% : Percentage of team offensive rebounds
    df_tmp['Team1_ORP'] = df_tmp.apply(lambda row: row.Team1_OR / (row.Team1_OR + row.Team2_DR), axis=1)
    df_tmp['Team2_ORP'] = df_tmp.apply(lambda row: row.Team2_OR / (row.Team2_OR + row.Team1_DR), axis=1)
    #DREB% : Percentage of team defensive rebounds
    df_tmp['Team1_DRP'] = df_tmp.apply(lambda row: row.Team1_DR / (row.Team1_DR + row.Team2_OR), axis=1)
    df_tmp['Team2_DRP'] = df_tmp.apply(lambda row: row.Team2_DR / (row.Team2_DR + row.Team1_OR), axis=1)                                      
    #REB% : Percentage of team total rebounds
    df_tmp['Team1_RP'] = df_tmp.apply(lambda row: (row.Team1_DR + row.Team1_OR) / (row.Team1_DR + row.Team1_OR + row.Team2_DR + row.Team2_OR), axis=1)
    df_tmp['Team2_RP'] = df_tmp.apply(lambda row: (row.Team2_DR + row.Team2_OR) / (row.Team1_DR + row.Team1_OR + row.Team2_DR + row.Team2_OR), axis=1) 

    tmp1 = df_tmp.apply(lambda row: row.Team1_Pts + row.Team1_FGM + row.Team1_FTM - row.Team1_FGA - row.Team1_FTA + row.Team1_DR + 0.5*row.Team1_OR + row.Team1_Ast +row.Team1_Stl + 0.5*row.Team1_Blk - row.Team1_PF - row.Team1_TO, axis=1)
    tmp2 = df_tmp.apply(lambda row: row.Team2_Pts + row.Team2_FGM + row.Team2_FTM - row.Team2_FGA - row.Team2_FTA + row.Team2_DR + 0.5*row.Team2_OR + row.Team2_Ast +row.Team2_Stl + 0.5*row.Team2_Blk - row.Team2_PF - row.Team2_TO, axis=1) 
    df_tmp['Team1_PIE'] = tmp1/(tmp1 + tmp2)
    df_tmp['Team2_PIE'] = tmp2/(tmp1 + tmp2)

    cols_simple_1 = ['Team1_FGM', 'Team1_FGA', 'Team1_FGM3', 'Team1_FGA3', 'Team1_FTM', 'Team1_FTA', 'Team1_OR', 'Team1_DR', 'Team1_Ast', 'Team1_TO', 'Team1_Stl', 'Team1_Blk', 'Team1_PF']
    cols_simple_2 = ['Team2_FGM', 'Team2_FGA', 'Team2_FGM3', 'Team2_FGA3', 'Team2_FTM', 'Team2_FTA', 'Team2_OR', 'Team2_DR', 'Team2_Ast', 'Team2_TO', 'Team2_Stl', 'Team2_Blk', 'Team2_PF']
    df_tmp.drop(cols_simple_1, axis=1, inplace=True)
    df_tmp.drop(cols_simple_2, axis=1, inplace=True)
    
    #cols_advanced_1 = ['Team1_Pts', 'Pos1', 'Team1_OffRtg', 'Team1_DefRtg', 'Team1_NetRtg', 'Team1_AstR', 'Team1_TOR', 'Team1_TSP', 'Team1_eFGP', 'Team1_FTAR', 'Team1_ORP', 'Team1_DRP', 'Team1_RP', 'Team1_PIE']
    #cols_advanced_2 = ['Team2_Pts', 'Pos2', 'Team2_OffRtg', 'Team2_DefRtg', 'Team2_NetRtg', 'Team2_AstR', 'Team2_TOR', 'Team2_TSP', 'Team2_eFGP', 'Team2_FTAR', 'Team2_ORP', 'Team2_DRP', 'Team2_RP', 'Team2_PIE']
    #cols_advanced_1 = ['Team1_Pts', 'Team1_NetRtg', 'Team1_AstR', 'Team1_TOR', 'Team1_TSP', 'Team1_eFGP', 'Team1_RP', 'Team1_PIE']
    #cols_advanced_2 = ['Team2_Pts', 'Team2_NetRtg', 'Team2_AstR', 'Team2_TOR', 'Team2_TSP', 'Team2_eFGP', 'Team2_RP', 'Team2_PIE']
    cols_advanced_1 = ['Team1_Pts', 'Team1_PIE']
    cols_advanced_2 = ['Team2_Pts', 'Team2_PIE']
    
    df_tmp_team1 = df_tmp[['TeamID1','Season'] + cols_advanced_1]
    df_tmp_team2 = df_tmp[['TeamID2','Season'] + cols_advanced_2]
    #MEAN IN ALL HIST DATA
    df_tmp_team1 = df_tmp_team1.merge(df_tmp_team1.groupby(['TeamID1'], as_index=False)[cols_advanced_1].mean(), how='left', on=['TeamID1'], suffixes=['',MEAN_SUFFIX])
    df_tmp_team2 = df_tmp_team2.merge(df_tmp_team2.groupby(['TeamID2'], as_index=False)[cols_advanced_2].mean(), how='left', on=['TeamID2'], suffixes=['',MEAN_SUFFIX])
    cols_advanced_1_mean = [c+MEAN_SUFFIX for c in cols_advanced_1]
    cols_advanced_2_mean = [c+MEAN_SUFFIX for c in cols_advanced_2]

    #LAST 2YEARS
    df_tmp_2years_team1 = df_tmp_team1.loc[df_tmp_team1['Season'].isin([season-1,season-2])]
    df_tmp_2years_team2 = df_tmp_team2.loc[df_tmp_team2['Season'].isin([season-1,season-2])]
    df_tmp_team1 = df_tmp_team1.merge(df_tmp_2years_team1[['TeamID1']+cols_advanced_1], how='left', on=['TeamID1'], suffixes=['',MEAN_2YRS_SUFFIX])
    df_tmp_team2 = df_tmp_team2.merge(df_tmp_2years_team2[['TeamID2']+cols_advanced_2], how='left', on=['TeamID2'], suffixes=['',MEAN_2YRS_SUFFIX])
    cols_advanced_1_last_2 = [c+MEAN_2YRS_SUFFIX for c in cols_advanced_1]
    cols_advanced_2_last_2 = [c+MEAN_2YRS_SUFFIX for c in cols_advanced_2]

    #LAST YEAR
    df_tmp_lastyear_team1 = df_tmp_team1.loc[df_tmp_team1['Season'] == season-1]
    df_tmp_lastyear_team2 = df_tmp_team2.loc[df_tmp_team2['Season'] == season-1]
    df_tmp_team1 = df_tmp_team1.merge(df_tmp_lastyear_team1[['TeamID1']+cols_advanced_1], how='left', on=['TeamID1'], suffixes=['',LAST_SUFFIX])
    df_tmp_team2 = df_tmp_team2.merge(df_tmp_lastyear_team2[['TeamID2']+cols_advanced_2], how='left', on=['TeamID2'], suffixes=['',LAST_SUFFIX])
    cols_advanced_1_last = [c+LAST_SUFFIX for c in cols_advanced_1]
    cols_advanced_2_last = [c+LAST_SUFFIX for c in cols_advanced_2]

    df_tmp_team1.drop(cols_advanced_1, axis=1, inplace=True)
    df_tmp_team1 = df_tmp_team1.drop_duplicates(subset=['TeamID1'])
    df_tmp_team2.drop(cols_advanced_2, axis=1, inplace=True) 
    df_tmp_team2 = df_tmp_team2.drop_duplicates(subset=['TeamID2'])
    
    #ORDINALS
    ordinals = df_ordinals_filtered[df_ordinals_filtered['Season']==season].groupby(['TeamID'], as_index=False)['OrdinalRank'].mean()
    ordinals_1 = ordinals.rename(columns={'OrdinalRank': 'Team1_OrdinalRank','TeamID': 'TeamID1'})
    df_tmp_team1 = df_tmp_team1.merge(ordinals_1, how='left', on=['TeamID1'])
    ordinals_2 = ordinals.rename(columns={'OrdinalRank': 'Team2_OrdinalRank','TeamID': 'TeamID2'})
    df_tmp_team2 = df_tmp_team2.merge(ordinals_2, how='left', on=['TeamID2'])
    cols_ordinals_1 = ['Team1_OrdinalRank']
    cols_ordinals_2 = ['Team2_OrdinalRank']


    #######MERGE!
    feature_data = feature_data.merge(df_tmp_team1[['TeamID1'] + cols_advanced_1_mean+cols_advanced_1_last+cols_advanced_1_last_2+cols_ordinals_1], how='left', on=['TeamID1'])
    feature_data = feature_data.merge(df_tmp_team2[['TeamID2'] + cols_advanced_2_mean+cols_advanced_2_last+cols_advanced_2_last_2+cols_ordinals_2], how='left', on=['TeamID2'])
    #Fill Na's
    for col in cols_advanced_1_mean+cols_advanced_2_mean+cols_advanced_1_last+cols_advanced_2_last+cols_advanced_1_last_2+cols_advanced_2_last_2+cols_ordinals_1+cols_ordinals_2:
        feature_data[col] = feature_data[col].fillna(feature_data[col].mean()) #probably naive

    ####################################################### REMOVE TEAM IDs #######################################################
    feature_data = feature_data.drop(['TeamID1','TeamID2'], 1)

    return feature_data

def renameColumns(df_concat, isWinner):

    if isWinner:
        df_concat = df_concat.rename( columns = { 'WTeamID' : 'TeamID1', 
                                                       'LTeamID' : 'TeamID2',
                                                      'WScore' : 'Team1_Score',
                                                      'LScore' : 'Team2_Score',
                                                      'WSeed' : 'Team1_Seed',
                                                      'LSeed' : 'Team2_Seed',
                                                      'WFGM' : 'Team1_FGM',
                                                      'LFGM' : 'Team2_FGM',
                                                      'WFGA' : 'Team1_FGA',
                                                      'LFGA' : 'Team2_FGA',
                                                      'WFGM3' : 'Team1_FGM3',
                                                      'LFGM3' : 'Team2_FGM3',
                                                      'WFGA3' : 'Team1_FGA3',
                                                      'LFGA3' : 'Team2_FGA3',
                                                      'WFTM' : 'Team1_FTM',
                                                      'LFTM' : 'Team2_FTM',
                                                      'WFTA' : 'Team1_FTA',
                                                      'LFTA' : 'Team2_FTA',
                                                      'WOR' : 'Team1_OR',
                                                      'LOR' : 'Team2_OR',
                                                      'WDR' : 'Team1_DR',
                                                      'LDR' : 'Team2_DR',
                                                      'WAst' : 'Team1_Ast',
                                                      'LAst' : 'Team2_Ast',
                                                      'WTO' : 'Team1_TO',
                                                      'LTO' : 'Team2_TO',
                                                      'WStl' : 'Team1_Stl',
                                                      'LStl' : 'Team2_Stl',
                                                      'WBlk' : 'Team1_Blk',
                                                      'LBlk' : 'Team2_Blk',
                                                      'WPF' : 'Team1_PF',
                                                      'LPF' : 'Team2_PF',
                                                      'WLoc' : 'Team1_Loc'})
    else:
        df_concat = df_concat.rename( columns = { 'WTeamID' : 'TeamID2', 
                                                       'LTeamID' : 'TeamID1',
                                                      'WScore' : 'Team2_Score',
                                                      'LScore' : 'Team1_Score',
                                                      'WSeed' : 'Team2_Seed',
                                                      'LSeed' : 'Team1_Seed',
                                                      'WFGM' : 'Team2_FGM',
                                                      'LFGM' : 'Team1_FGM',
                                                      'WFGA' : 'Team2_FGA',
                                                      'LFGA' : 'Team1_FGA',
                                                      'WFGM3' : 'Team2_FGM3',
                                                      'LFGM3' : 'Team1_FGM3',
                                                      'WFGA3' : 'Team2_FGA3',
                                                      'LFGA3' : 'Team1_FGA3',
                                                      'WFTM' : 'Team2_FTM',
                                                      'LFTM' : 'Team1_FTM',
                                                      'WFTA' : 'Team2_FTA',
                                                      'LFTA' : 'Team1_FTA',
                                                      'WOR' : 'Team2_OR',
                                                      'LOR' : 'Team1_OR',
                                                      'WDR' : 'Team2_DR',
                                                      'LDR' : 'Team1_DR',
                                                      'WAst' : 'Team2_Ast',
                                                      'LAst' : 'Team1_Ast',
                                                      'WTO' : 'Team2_TO',
                                                      'LTO' : 'Team1_TO',
                                                      'WStl' : 'Team2_Stl',
                                                      'LStl' : 'Team1_Stl',
                                                      'WBlk' : 'Team2_Blk',
                                                      'LBlk' : 'Team1_Blk',
                                                      'WPF' : 'Team2_PF',
                                                      'WLoc' : 'Team2_Loc'})
    return df_concat

#the seed information
df_seeds = pd.read_csv('../input/NCAATourneySeeds.csv')
#tour information
df_tour = pd.read_csv('../input/NCAATourneyDetailedResults.csv')
#regular tourney info
df_reg = pd.read_csv('../input/RegularSeasonDetailedResults.csv')
#pd.read_csv('../input/MasseyOrdinals.csv')
#MasseyOrdinals_Prelim2018.csv
df_ordinals = pd.read_csv('../input/MasseyOrdinals_thruSeason2018_Day128.csv')
#remove systems without the full 351 teams to make our lives easier
df_ordinals_filtered= df_ordinals.groupby(['SystemName']).filter(lambda x: x['OrdinalRank'].max() == 351).reset_index(drop=True)
#df_ordinals_filtered.groupby(['SystemName'])['OrdinalRank'].max().unique()

#recover seed integer (remove region data)
df_seeds['seed_int'] = df_seeds['Seed'].apply( lambda x : int(x[1:3]) )

#create win/loss seeds and add that info to tour
df_winseeds = df_seeds.loc[:, ['TeamID', 'Season', 'seed_int']].rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.loc[:, ['TeamID', 'Season', 'seed_int']].rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])

#do the same with reg
df_dummy_reg = pd.merge(left=df_reg, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat_reg = pd.merge(left=df_dummy_reg, right=df_lossseeds, on=['Season', 'LTeamID'])
#fill seed NaN's with worse value+1
#df_concat_reg['WSeed'] = df_concat_reg['WSeed'].fillna(value=max_seed+1)
#df_concat_reg['LSeed'] = df_concat_reg['LSeed'].fillna(value=max_seed+1)
#fill seed NaN's with average value
df_concat_reg['WSeed'] = df_concat_reg['WSeed'].fillna(default_seed)
df_concat_reg['LSeed'] = df_concat_reg['LSeed'].fillna(default_seed)

#prepares sample submission
df_sample_sub = pd.read_csv('../input/SampleSubmissionStage2.csv')
df_sample_sub['Season'] = df_sample_sub['ID'].apply(lambda x : int(x.split('_')[0]) )
df_sample_sub['TeamID1'] = df_sample_sub['ID'].apply(lambda x : int(x.split('_')[1]) )
df_sample_sub['TeamID2'] = df_sample_sub['ID'].apply(lambda x : int(x.split('_')[2]) )
df_sample_sub['Team1_Seed'] = df_sample_sub.merge(df_seeds, how = 'left', left_on = 'TeamID1', right_on = 'TeamID')['seed_int']
df_sample_sub['Team2_Seed'] = df_sample_sub.merge(df_seeds, how = 'left', left_on = 'TeamID2', right_on = 'TeamID')['seed_int']

#create trainign data
winners = renameColumns(df_concat, True)
winners['Result'] = 1.0
losers = renameColumns(df_concat, False)
losers['Result'] = 0.0

winners_reg = renameColumns(df_concat_reg, True)
winners_reg['Result'] = 1.0
losers_reg = renameColumns(df_concat_reg, False)
losers_reg['Result'] = 0.0

#aggregate losers and winners data for regular and ncaa season data
ncaa_data = pd.concat( [winners, losers], axis = 0).reset_index(drop = True)
reg_data = pd.concat( [winners_reg, losers_reg], axis = 0).reset_index(drop = True)
#hist_data = pd.concat( [ncaa_data, ncaa_data], axis = 0).reset_index(drop = True)
#hist_data = ncaa_data
hist_data = pd.concat( [ncaa_data, reg_data], axis = 0).reset_index(drop = True)

#prepare training data for feature creation
train_x = pd.DataFrame()
train_y = pd.Series()

#tests using 2017 as validation data show that going back further than train_year-1 makes results worse. Dunno why, maybe the teams change too much?
for x in range(train_year-train_offset, train_year+1):
#for x in range(train_first_year, train_year+1):
    train_current_season_tmp = hist_data.loc[hist_data['Season'] == x]
    train_x_tmp = createFeatures(hist_data, x, train_first_year, train_current_season_tmp, df_ordinals_filtered)
    train_y_tmp = np.ravel(train_current_season_tmp['Result'])
    
    train_x = pd.concat([train_x, train_x_tmp], axis = 0)
    train_y = np.concatenate([train_y, train_y_tmp], axis = 0)
#train_x = train_x.drop(['TeamID1','TeamID2','Result','Team1_Score','Team2_Score','NumOT','Season','DayNum'], 1)

#prepare prediction data
if IS_VALIDATION:
    predict_current_season = ncaa_data.loc[ncaa_data['Season'] == predict_year]
    predict_y = np.ravel(predict_current_season['Result'])
else:
    predict_current_season = df_sample_sub
predict_x = createFeatures(hist_data, predict_year, train_first_year, predict_current_season, df_ordinals_filtered)
#remove columns with failure tolerance - real prediction x will not have results
#cols = ['ID','Pred','TeamID1','TeamID2','Result','Team1_Score','Team2_Score','NumOT','Season','DayNum']
#cols = [c for c in cols if c in predict_x.columns]
#predict_x = predict_x.drop(cols,1)
#predict_x = predict_x.drop(['TeamID1','TeamID2','Result','Team1_Score','Team2_Score','NumOT','Season','DayNum'], 1)

train_x_max = train_x.max()
train_x_min = train_x.min()
train_x = ( train_x - train_x_min ) / ( train_x_max - train_x_min + 1e-14)

predict_x_max = predict_x.max()
predict_x_min = predict_x.min()
predict_x = ( predict_x - predict_x_min ) / ( predict_x_max - predict_x_min + 1e-14)


model = LogisticRegressionCV(cv = 5)
model = model.fit(train_x, train_y)
#train_x.columns
#model.coef_
#print(zip(train_x.columns, model.coef_))

#model.score(predict_x, predict_y)
if IS_VALIDATION:
    prediction = model.predict_proba(predict_x)
    loss = metrics.log_loss(predict_y, prediction)
    print(prediction[1:20])
    print(loss)
else:
    predict_current_season['Pred'] = model.predict_proba(predict_x)
    predict_current_season = predict_current_season[['ID','Pred']]
    predict_current_season.to_csv('predict.csv', index = False)
    print(predict_current_season.head(20))
#get results for test
