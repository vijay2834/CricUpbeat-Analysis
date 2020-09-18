import streamlit as st
import pandas as pd
import numpy as np
import plotly.plotly as py
from plotly import tools
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=False)
import plotly.figure_factory as ff
import plotly.graph_objs as go

import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder



st.title("CricUpbeat Statistics")

st.sidebar.title("📈Statictics")
predict = st.sidebar.button("🏆Win Predictor")
bat_bar = st.sidebar.button("🏏Batsman Analysis")
bowl_bar = st.sidebar.button("🥎Bowler Analysis")
field_bar = st.sidebar.button("🧤Fielder Analysis")
team  = st.sidebar.button('🏟Match Analysis')
runs_val = st.sidebar.button("💯Score Analysis")
powerplay = st.sidebar.button("✋Powerplay Analysis")
avg = st.sidebar.button("📊Average Score Analysis")
fantasy = st.sidebar.button("🎯Fantasy team best players")


predict_main = st.button("🏆WIN PREDICTOR")
bat_bar_main = st.button("🏏BATSMAN ANALYSIS")
bowl_bar_main = st.button("🥎BOWLER ANALYSIS")
field_bar_main = st.button("🧤FIELDER ANALYSIS")
team_main  = st.button('🏟MATCH ANALYSIS')
runs_val_main = st.button("💯SCORE ANALYSIS")
powerplay_main = st.button("✋POWERPLAY ANALYSIS")
avg_main = st.button("📊AVERAGE SCORE ANALYSIS")
fantasy_main = st.button("🎯FANTASY TEAM BEST PLAYERS ")

@st.cache(persist = True,allow_output_mutation=True)
def data():
    df = pd.read_csv(r'https://raw.githubusercontent.com/vijay2834/IPL-data/master/deliveries.csv')
    mat = pd.read_csv(r'https://raw.githubusercontent.com/vijay2834/IPL-data/master/matches.csv')

    return df,mat
deliveries,matches = data()

s_man_of_match = (matches.groupby(matches.player_of_match).player_of_match.count().
              sort_values(ascending=False).head(15))

df_man_of_match =(s_man_of_match.to_frame().rename
                  (columns = {"player_of_match": "times"}).reset_index())

cen = deliveries.groupby(['batsman','match_id']).agg({'batsman_runs':'sum'})
cen = cen[cen['batsman_runs']>=100]
cen = cen.groupby(['batsman']).agg({'count'})
cen.columns = cen.columns.droplevel()
cen = cen.sort_values(by='count',ascending=False).reset_index()

half_cen = deliveries.groupby(['batsman','match_id']).agg({'batsman_runs':'sum'})
half_cen = half_cen[half_cen['batsman_runs']>=50]
half_cen = half_cen[half_cen['batsman_runs']<100]
half_cen = half_cen.groupby(['batsman']).agg({'count'})
half_cen.columns = half_cen.columns.droplevel()
half_cen = half_cen.sort_values(by='count',ascending=False).reset_index()

df_big = pd.merge(cen,half_cen, on='batsman',how='right')
df_big = df_big.fillna(0)

df_strike_rate = deliveries.groupby(['batsman']).agg({'ball':'count','batsman_runs':'mean'}).sort_values(by='batsman_runs',ascending=False)
df_strike_rate.rename(columns ={'batsman_runs' : 'strike rate'}, inplace=True)

df_runs_per_match = deliveries.groupby(['batsman','match_id']).agg({'batsman_runs':'sum'})
df_total_runs = df_runs_per_match.groupby(['batsman']).agg({'sum' ,'mean','count'})
df_total_runs.rename(columns ={'sum' : 'batsman run','count' : 'match count','mean' :'average score'}, inplace=True)
df_total_runs.columns = df_total_runs.columns.droplevel()

df_sixes = deliveries[['batsman','batsman_runs']][deliveries.batsman_runs==6].groupby(['batsman']).agg({'batsman_runs':'count'})
df_four = deliveries[['batsman','batsman_runs']][deliveries.batsman_runs==4].groupby(['batsman']).agg({'batsman_runs':'count'})

df_batsman_stat = pd.merge(pd.merge(pd.merge(df_strike_rate,df_total_runs, left_index=True, right_index=True),
                                    df_sixes, left_index=True, right_index=True),df_four, left_index=True, right_index=True)

df_batsman_stat.rename(columns = {'ball' : 'Ball', 'strike rate':'Strike Rate','batsman run' : 'Batsman Run','match count' : 'Match Count',
                                  'average score' : 'Average score' ,'batsman_runs_x' :'Six','batsman_runs_y':'Four'},inplace=True)
df_batsman_stat['Strike Rate'] = df_batsman_stat['Strike Rate']*100
df_batsman_stat = df_batsman_stat.sort_values(by='Batsman Run',ascending=False).reset_index()

batsman_stats = pd.merge(df_batsman_stat,df_big, on='batsman',how='left').fillna(0)
batsman_stats.rename(columns = {'count_x' : '100s', 'count_y' : '50s'},inplace=True)

condition_catch = (deliveries.dismissal_kind == 'caught')
condition_run= (deliveries.dismissal_kind == 'run out')
condition_stump= (deliveries.dismissal_kind == 'stumped')
condition_caught_bowled = (deliveries.dismissal_kind == 'caught and bowled')

s_catch = deliveries.loc[condition_catch,:].groupby(deliveries.fielder).dismissal_kind.count().sort_values(ascending=False)
s_run = deliveries.loc[condition_run,:].groupby(deliveries.fielder).dismissal_kind.count().sort_values(ascending=False)
s_stump = deliveries.loc[condition_stump,:].groupby(deliveries.fielder).dismissal_kind.count().sort_values(ascending=False)
s_caught_bowled = deliveries.loc[condition_caught_bowled,:].groupby(deliveries.bowler).dismissal_kind.count().sort_values(ascending=False)

df_catch= s_catch.to_frame().reset_index().rename(columns ={'dismissal_kind' : 'catch'})
df_run= s_run.to_frame().reset_index().rename(columns ={'dismissal_kind' : 'run_out'})
df_stump= s_stump.to_frame().reset_index().rename(columns ={'dismissal_kind' : 'stump'})
df_caught_bowled = s_caught_bowled.to_frame().reset_index().rename(columns ={'dismissal_kind' : 'caught and bowled'})

df_field = pd.merge(pd.merge(df_catch,df_run,on='fielder', how='outer'),df_stump,on='fielder',how='outer')
field_stats = df_field[~df_field['fielder'].str.contains("(sub)")].reset_index().drop(['index'],axis=1).fillna(0)

condition = ((deliveries.dismissal_kind.notnull()) &(deliveries.dismissal_kind != 'run out')&
            (deliveries.dismissal_kind != 'retired hurt' )&(deliveries.dismissal_kind != 'hit wicket')
            &(deliveries.dismissal_kind != 'obstructing the field')&(deliveries.dismissal_kind != 'caught and bowled'))

df_bowlers = deliveries.loc[condition,:].groupby(deliveries.bowler).dismissal_kind.count().sort_values(ascending=False).reset_index()
df_bowlers = pd.merge(df_bowlers,df_caught_bowled , on='bowler',how='left').fillna(0)

high=deliveries.groupby(['match_id', 'bowler']).agg({'total_runs':'sum'}).reset_index()

over_count=deliveries.groupby(['match_id', 'bowler','over']).agg({'total_runs':'sum'}).reset_index()
overs = over_count.groupby(['match_id','bowler']).agg({'over':'count'}).reset_index()
overs = overs[overs['over']>=2]

bowlers = pd.merge(high,overs,on=['match_id', 'bowler'], how='right')
bowlers['economy'] = bowlers['total_runs']/bowlers['over']
bowlers['eco_range'] = pd.cut(bowlers['economy'], [0, 4, 5, 6, 9, 10, 11, 30], labels=['below4', '4-5', '5-6', '6-9','9-10','10-11','above11'])

bowlers = pd.concat([bowlers,pd.get_dummies(bowlers['eco_range'], prefix='eco')],axis=1)
economy_rates=bowlers.groupby(['bowler']).agg({'eco_below4':'sum','eco_4-5':'sum','eco_5-6':'sum','eco_6-9':'sum','eco_9-10':'sum','eco_10-11':'sum','eco_above11':'sum'}).reset_index()

maiden_over = over_count[over_count['total_runs']==0]
maidens = maiden_over['bowler'].value_counts().to_frame().reset_index().rename({'index':'bowler','bowler':'maiden_overs'},axis=1)

hauls=deliveries.groupby(['match_id', 'bowler']).agg({'player_dismissed':'count'}).reset_index()
hauls = hauls[hauls['player_dismissed']>=4]
hauls['haul'] = pd.cut(hauls['player_dismissed'], [0,4,8], labels=['4', '5'])
hauls = pd.concat([hauls,pd.get_dummies(hauls['haul'], prefix='haul')],axis=1)
hauls.drop(['player_dismissed','haul'],inplace=True,axis=1)
hauls=hauls.groupby(['bowler']).agg({'haul_4':'sum','haul_5':'sum'}).reset_index()

bowlers_stats = pd.merge(pd.merge(pd.merge(economy_rates,maidens,on='bowler', how='left'),df_bowlers,on='bowler',how='left'),hauls,on='bowler',how='right').fillna(0)
bowlers_stats.rename(columns ={'dismissal_kind' : 'wickets'},inplace=True)
centuries = batsman_stats.sort_values(by='100s').tail(15)
half_centuries = batsman_stats.sort_values(by='50s').tail(15)

cen = batsman_stats[['100s','50s','batsman']]
cen['points'] = (cen['100s']*8) + (cen['50s']*4)
cen.sort_values(by='points',inplace=True,ascending=False)

fours = batsman_stats.sort_values(by='Four').tail(15)
sixes = batsman_stats.sort_values(by='Six').tail(15)
runs = batsman_stats.sort_values(by='Batsman Run').tail(15)

runs = batsman_stats[['Six','Four','Batsman Run','batsman']]
runs['point'] = (runs['Six']*1) + (runs['Four']*0.5) + (runs['Batsman Run']*0.5)
runs.sort_values(by='point',inplace=True,ascending=False)

final = pd.merge(cen,runs,on='batsman', how='inner')
final['total_points']=final['points']+final['point']
final['max'] = final['100s']+final['50s']

final.sort_values(by='total_points',ascending=False,inplace=True)
best_batsman = final[['batsman','total_points']]

final['Batsman Run'] = (final['Batsman Run'])/(final['Batsman Run'].max()/100)
final['Six'] = (final['Six'])/(final['Six'].max()/100)
final['Four'] = (final['Four'])/(final['Four'].max()/100)
final['max'] = (final['max'])/(final['max'].max()/100)
final['total_points'] = (final['total_points'])/(final['total_points'].max()/100)

x = final[final["batsman"] == "V Kohli"]
y = final[final["batsman"] == "RG Sharma"]
z = final[final["batsman"] == "SK Raina"]

field = field_stats[['fielder','stump','catch','run_out']]

field1 = field[(field['stump'] > 0)]
field2 = field[~(field['stump'] > 0)]

field1['points'] = (field1['catch']*4) + (field1['stump']*6) + (field1['run_out']*2)
field2['points'] = (field2['catch']*4) + (field2['stump']*6) + (field2['run_out']*6)

field = pd.concat([field1, field2])
field.sort_values(by='points',ascending=False,inplace=True)

field1.sort_values(by='points',ascending=False,inplace=True)
field2.sort_values(by='points',ascending=False,inplace=True)

best_fielder = field[['fielder','points']]

haul5 = bowlers_stats.sort_values(by='haul_5',ascending=False).head(10)
haul4 = bowlers_stats.sort_values(by='haul_4',ascending=False).head(10)

wicket = bowlers_stats.sort_values(by='wickets',ascending=False).head(10)
caught_bowled = bowlers_stats.sort_values(by='caught and bowled',ascending=False).head(10)

dismissals = bowlers_stats[['bowler','wickets','caught and bowled']]
dismissals['dismissals'] = dismissals['wickets']+dismissals['caught and bowled']

dismissals['points'] = (dismissals['wickets']*10) + (dismissals['caught and bowled']*14)
dismissals.sort_values(by='points',ascending=False,inplace=True)

e1 = bowlers_stats.sort_values(by='eco_below4',ascending=False).head(10)
e2 = bowlers_stats.sort_values(by='eco_4-5',ascending=False).head(10)
e3 = bowlers_stats.sort_values(by='eco_5-6',ascending=False).head(10)
e4 = bowlers_stats.sort_values(by='eco_6-9',ascending=False).head(10)
e5 = bowlers_stats.sort_values(by='eco_9-10',ascending=False).head(10)
e6 = bowlers_stats.sort_values(by='eco_10-11',ascending=False).head(10)
e7 = bowlers_stats.sort_values(by='eco_above11',ascending=False).head(10)
m = bowlers_stats.sort_values(by='maiden_overs',ascending=False).head(10)

eco = bowlers_stats[['bowler','maiden_overs','eco_below4','eco_4-5','eco_5-6','eco_9-10','eco_10-11','eco_above11']]

eco['points'] = ((eco['eco_below4']*3)+(eco['eco_4-5']*2)+(eco['eco_5-6']*1)+
                 (eco['eco_9-10']*(-1))+(eco['eco_10-11']*(-2))+(eco['eco_above11']*(-3))+(eco['maiden_overs']*4))

eco.sort_values(by='points',ascending=False,inplace=True)

final = bowlers_stats
final['points_x'] = ((final['eco_below4']*3)+(final['eco_4-5']*2)+(final['eco_5-6']*1)+(final['eco_9-10']*(-1))+
                   (final['eco_10-11']*(-2))+(final['eco_above11']*(-3))+(final['maiden_overs']*4))

final['points_y'] = (final['wickets']*10) + (final['caught and bowled']*14)
final['points_z'] = (final['haul_4']*4) + (final['haul_5']*8)

final['points'] = final['points_x']+final['points_y']+final['points_z']
final['dismissals'] = final['wickets']+final['caught and bowled']

final.sort_values(by='points',ascending=False,inplace=True)
final_bowl = final.head(10)

best_bowler = final[['bowler','points']]

final['points_x'] = (final['points_x'])/(final['points_x'].max()/100)
final['points_y'] = (final['points_y'])/(final['points_y'].max()/100)
final['points_z'] = (final['points_z'])/(final['points_z'].max()/100)
final['points'] = (final['points'])/(final['points'].max()/100)

x_bowl = final[final["bowler"] == "Harbhajan Singh"]
y_bowl = final[final["bowler"] == "SP Narine"]
z_bowl = final[final["bowler"] == "R Ashwin"]
w_bowl = final[final["bowler"] == "B Kumar"]

season_winner=matches.drop_duplicates(subset=['season'], keep='last')[['season','winner']].reset_index(drop=True)
season_winner = season_winner['winner'].value_counts()

season_winner = season_winner.to_frame()
season_winner.reset_index(inplace=True)
season_winner.rename(columns={'index':'team'},inplace=True)
finals=matches.drop_duplicates(subset=['season'],keep='last')
finals=finals[['id','season','city','team1','team2','toss_winner','toss_decision','winner']]
most_finals=pd.concat([finals['team1'],finals['team2']]).value_counts().reset_index()
most_finals.rename({'index':'team',0:'count'},axis=1,inplace=True)
xyz=finals['winner'].value_counts().reset_index()

most_finals=most_finals.merge(xyz,left_on='team',right_on='index',how='outer')
most_finals=most_finals.replace(np.NaN,0)
most_finals.drop('index',axis=1,inplace=True)
most_finals.set_index('team',inplace=True)
most_finals.rename({'count':'finals_played','winner':'won_count'},inplace=True,axis=1)
most_finals.reset_index(inplace=True)


def pre(matches,team1,team2,ven):
    df = matches[['team1','team2','toss_decision','winner']]
    df['winner'].fillna(df['team1'],inplace = True)
    lbl = LabelEncoder()
    rd = RandomForestClassifier()
    x = df[['team1','team2','toss_decision']]
    col  = ['team1','team2','toss_decision']
    mapping = []
    for val in col:
        x[val] = lbl.fit_transform(x[val])
        mapping.append(dict(zip(lbl.classes_,lbl.transform(lbl.classes_))))
    y = df[['winner']]
    v = {'Home':0,'Away':1}
    rd.fit(x,y)

    a = mapping[0][team1]
    b = mapping[1][team2]
    c = v[ven]
    k = rd.predict([[a,b,c]])
    z = rd.score(x,y)
    return k


def authenticate(username, password):
    return username == "buddha" and password == "s4msara"

import sys



st.title("🏆Win Predictor")
val = np.asarray(['CSK','DC','KKR','MI','RCB','SRH','RR','KXIP'],dtype = object)

team1 = st.selectbox("Select Team 1",val)
if(team1):
    team2 = st.selectbox("Select Team 2",np.delete(val,list(val).index(team1)))
    if(team2):
        venue = st.selectbox("Select Venue",np.asarray(['Home','Away']))
        if(venue):
            sc = st.button("Predict")
            if(sc):
                winner = pre(matches,team1,team2,venue)
                st.write("The win predictor predicted "+ winner[0]+" as winner with 67% chances.")


if(predict or predict_main):
    st.title("Winner Analysis")
    st.markdown("Teams with highest No. of Trophies")
    trace0 = go.Pie(labels=season_winner['team'], values=season_winner['winner'],
                  hoverinfo='label+value+name',name="Winner")

    layout=go.Layout(title='Winner of IPL season')
    fig = go.Figure(data=[trace0], layout=layout)
    st.plotly_chart(fig,use_container_width=True)
    trace1 = go.Bar(x=most_finals.team,y=most_finals.finals_played,
                    name='Total Matches',opacity=0.4)

    trace2 = go.Bar(x=most_finals.team,y=most_finals.won_count,
                    name='Matches Won',marker=dict(color='red'),opacity=0.4)

    data = [trace1, trace2]

    layout = go.Layout(title='Match Played vs Wins In Finals',xaxis=dict(title='Team'),
                       yaxis=dict(title='Count'),bargap=0.2,bargroupgap=0.1)

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,use_container_width=True)
    df=finals[finals['toss_winner']==finals['winner']]
    slices=[len(df),(len(finals)-len(df))]
    labels=['yes','no']
    trace0 = go.Pie(labels=labels, values=slices,
                  hoverinfo='label+value+name',name="Winner")

    layout=go.Layout(title='Winner of IPL season with toss in favour')
    fig = go.Figure(data=[trace0], layout=layout)
    st.plotly_chart(fig,use_container_width=True)




if(bat_bar or bat_bar_main):
    data = [go.Bar(x=df_man_of_match['player_of_match'],
                   y=df_man_of_match["times"],
                   marker=dict(color='#b4122c'),opacity=0.75)]

    layout = go.Layout(title='Man of the Matches ',
                       xaxis=dict(title='Player',tickmode='linear'),
                       yaxis=dict(title='Count'),bargap=0.2)

    fig1 = go.Figure(data=data, layout=layout)
    st.title("Batsman Analysis")
    st.plotly_chart(fig1,use_container_width = True)

    st.markdown("Centuries and Half Centuries Analysis")

    fig2 = {"data" : [{"x" : centuries["batsman"],"y" : centuries["100s"],
                  "name" : "100s","marker" : {"color" : "#012154","size": 12},
                  "line": {"width" : 3},"type" : "scatter","mode" : "lines+markers" ,
                  "xaxis" : "x1","yaxis" : "y1"},

                 {"x" : half_centuries["batsman"],"y" : half_centuries["50s"],
                  "name" : "50s","marker" : {"color" : "#b4122c","size": 12},
                  "type" : "scatter","line": {"width" : 3},"mode" : "lines+markers",
                  "xaxis" : "x2","yaxis" : "y2"}],

        "layout" : {"title": "Total centuries and half-centuries by top batsman",
                    "xaxis2" : {"domain" : [0, 1],"anchor" : "y2",
                    "showticklabels" : True},"margin" : {"b" : 111},
                    "yaxis2" : {"domain" : [.55, 1],"anchor" : "x2","title": "50s"},
                    "xaxis" : {"domain" : [0, 1],"tickmode":'linear',"title": "Batsman"},
                    "yaxis" : {"domain" :[0, .45], "anchor" : "x2","title": "100s"}}}

    st.plotly_chart(fig2,use_container_width=True)




    trace = go.Table(
        domain=dict(x=[0, 0.55],
                    y=[0, 1.0]),
        header=dict(values=["Batsman","Points","100s","50s"],
                    fill = dict(color = 'pink'),
                    font = dict(color = 'white', size = 14),
                    align = ['center'],
                   height = 30),
        cells=dict(values=[cen['batsman'].head(10), cen['points'].head(10), cen['100s'].head(10), cen['50s'].head(10)],
                   fill = dict(color = ['pink', 'white']),
                   align = ['center']))

    trace1 = go.Bar(x=cen['batsman'].head(10),
                    y=cen["points"].head(10),
                    xaxis='x1',
                    yaxis='y1',
                    marker=dict(color='#012154'),opacity=0.60)

    layout = dict(
        width=830,
        height=415,
        autosize=False,
        title='Batsman with highest points by centuries and half centuries',
        margin = dict(t=100),
        showlegend=False,
        xaxis1=dict(**dict(domain=[0.65, 1], anchor='y1', showticklabels=True)),
        yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),
    )

    fig1 = dict(data=[trace, trace1], layout=layout)
    st.plotly_chart(fig1,use_container_width=True)
    st.markdown("Boundaries and total runs analysis")


    trace1 = go.Scatter(x=sixes.batsman,y =sixes.Six,name='6"s',marker =dict(color= "blue",size = 9),line=dict(width=2,dash='dash'),showlegend=True)
    trace2 = go.Scatter(x=fours.batsman,y = fours.Four,name='4"s',marker =dict(color= "green",size = 9),line=dict(width=2,dash='longdash'))
    trace3 = go.Scatter(x=runs.batsman,y = runs['Batsman Run'],name='2"s',marker =dict(color= "red",size = 9),line=dict(width=2,dash='dashdot'))

    fig = tools.make_subplots(rows=3, cols=1, subplot_titles=('Top 6"s Scorer','Top 4"s Scorer',"Highest total runs"), print_grid=False)

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)
    fig.append_trace(trace3, 3, 1)

    fig['layout'].update(height=700, width=820,title='Top Scorer in Boundaries and Total Runs',showlegend=False)
    st.plotly_chart(fig,use_container_width=True)



    trace = go.Table(
        domain=dict(x=[0, 0.55],
                    y=[0, 1.0]),
        header=dict(values=["Batsman","Points","Sixes","Fours"],
                    fill = dict(color='#012154'),
                    font = dict(color=['white'] * 5, size=14),
                    align = ['center'],
                   height = 30),
        cells=dict(values=[runs['batsman'].head(10), runs['point'].head(10), runs['Six'].head(10), runs['Four'].head(10)],
                   fill = dict(color=['#012154', 'white']),
                   align = ['center']))

    trace1 = go.Bar(x=runs['batsman'].head(10),
                    y=runs["point"].head(10),
                    xaxis='x1',
                    yaxis='y1',
                    marker=dict(color='#b4122c'),opacity=0.60)

    layout = dict(
        width=830,
        height=415,
        autosize=False,
        title='Batsman with highest points by runs and boundaries',
        margin = dict(t=100),
        showlegend=False,
        xaxis1=dict(**dict(domain=[0.61, 1], anchor='y1', showticklabels=True)),
        yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),
    )

    fig1 = dict(data=[trace, trace1], layout=layout)
    st.plotly_chart(fig1,use_container_width=True)


    data = [go.Scatterpolar(
      r = [x['Four'].values[0],x['Six'].values[0],x['Batsman Run'].values[0],x['max'].values[0],x['total_points'].values[0]],
      theta = ['Four','Six','Runs','Centuries','Points'],
      fill = 'toself', opacity = 0.8,
      name = "V Kohli"),

        go.Scatterpolar(
      r = [y['Four'].values[0],y['Six'].values[0],y['Batsman Run'].values[0],y['max'].values[0],y['total_points'].values[0]],
      theta = ['Four','Six','Runs','Centuries','Points'],
      fill = 'toself',subplot = "polar2",
        name = "RG Sharma"),

        go.Scatterpolar(
      r = [z['Four'].values[0],z['Six'].values[0],z['Batsman Run'].values[0],z['max'].values[0],z['total_points'].values[0]],
      theta = ['Four','Six','Runs','Centuries','Points'],
      fill = 'toself',subplot = "polar3",
        name = "SK Raina")]

    layout = go.Layout(title = "Comparison Between V Kohli, CH Gayle, S Raina",

                       polar = dict(radialaxis = dict(visible = True,range = [0, 100]),
                       domain = dict(x = [0, 0.25],y = [0, 1])),

                       polar2 = dict(radialaxis = dict(visible = True,range = [0, 100]),
                       domain = dict(x = [0.35, 0.65],y = [0, 1])),

                       polar3 = dict(radialaxis = dict(visible = True,range = [0, 100]),
                       domain = dict(x = [0.75, 1.0],y = [0, 1])),)

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,use_container_width = True)

if(field_bar or field_bar_main):
    st.title("Fielder Analysis")
    st.markdown("Fielderes with maximum catches")
    trace1 = go.Bar(x=field_stats.fielder.head(15),y=field_stats.catch,
                name='Caught',opacity=0.4)

    trace2 = go.Bar(x=field_stats.fielder.head(15),y=field_stats.run_out,name='Run out',
                    marker=dict(color='red'),opacity=0.4)

    trace3 = go.Bar(x=field_stats.fielder.head(15),y=field_stats.stump,name='Stump out',
                    marker=dict(color='lime'),opacity=0.4)

    data = [trace1, trace2, trace3]
    layout = go.Layout(title='Best fielders',
                       xaxis=dict(title='Player',tickmode='linear'),
                       yaxis=dict(title='Dismissals'),bargap=0.2,bargroupgap=0.1)

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,use_container_width = True)

    st.markdown("Fielder with maximum points")

    trace = go.Table(
        domain=dict(x=[0, 0.65],
                    y=[0, 1.0]),
        header=dict(values=["Fielder","Stump","Catch","Run out","Points"],
                    fill = dict(color = 'grey'),
                    font = dict(color = 'white', size = 14),
                    align = ['center'],
                   height = 30),
        cells=dict(values=[field['fielder'].head(10),field['stump'].head(10),field['catch'].head(10),field['run_out'].head(10),field['points'].head(10)],
                   fill = dict(color = ['lightgrey', 'white']),
                   align = ['center']))

    trace1 = go.Bar(x=field['fielder'].head(10),
                    y=field["points"].head(10),
                    xaxis='x1',
                    yaxis='y1',
                    marker=dict(color='hotpink'),opacity=0.60)

    layout = dict(
        width=850,
        height=440,
        autosize=False,
        title='Fielder with maximum Points',
        showlegend=False,
        xaxis1=dict(**dict(domain=[0.7, 1], anchor='y1', showticklabels=True)),
        yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),
    )

    fig1 = dict(data=[trace, trace1], layout=layout)
    st.plotly_chart(fig1,use_container_width=True)

    st.markdown("Fielders and wicketskeepers with maximum points")
    trace0 = go.Scatter(
        x=field1['points'].head(5),
        y=field1['fielder'],
        name = 'Wicketkeeper',
        mode='markers',
        marker=dict(
            color='rgba(156, 165, 196, 0.95)',
            line=dict(color='rgba(156, 165, 196, 1.0)',width=1),
            symbol='circle',
            size=16,
        ))
    trace1 = go.Scatter(
        x=field2['points'].head(5),
        y=field2['fielder'],
        name='Fielder',
        mode='markers',
        marker=dict(
            color='rgba(204, 204, 204, 0.95)',
            line=dict(color='rgba(217, 217, 217, 1.0)',width=1),
            symbol='circle',
            size=16,
        ))

    data = [trace0,trace1]
    layout = go.Layout(
        title="Ten best Fielders and Wicketkeepers for Fantasy League ",
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgb(102, 102, 102)',
            titlefont=dict(color='rgb(204, 204, 204)'),
            tickfont=dict(color='rgb(102, 102, 102)',),
            showticklabels=True,
            ticks='outside',
            tickcolor='rgb(102, 102, 102)',
        ),
        margin=dict(l=140,r=40,b=50,t=80),
        legend=dict(
            font=dict(size=10,),
            yanchor='middle',
            xanchor='right',
        ),
        hovermode='closest',
    )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,use_container_width = True)
if(bowl_bar or bowl_bar_main):
    st.title("Bowler Analysis")
    st.markdown("Wicket houls analysis")

    trace1 = go.Scatter(x=haul5['bowler'],y=haul5['haul_5'],name='5 Wickets Haul',marker =dict(color= "gold",size = 13),line=dict(width=3,dash='longdashdot'))
    trace2 = go.Scatter(x=haul4['bowler'],y=haul4['haul_4'],name='4 Wickets Haul',marker =dict(color= "lightgrey",size = 13),line=dict(width=3))

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Five Wickets','Four Wickets'), print_grid=False)

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)

    st.plotly_chart(fig,use_container_width = True)

    st.markdown("Bowlers with maximum dismissals analysis")

    trace = go.Table(
        domain=dict(x=[0, 0.52],
                    y=[0, 1.0]),
        header=dict(values=["Bowler","Dismissals","Points"],
                    fill = dict(color = 'red'),
                    font = dict(color = 'white', size = 14),
                    align = ['center'],
                   height = 30),
        cells=dict(values=[dismissals['bowler'].head(10),dismissals['dismissals'].head(10),dismissals['points'].head(10)],
                   fill = dict(color = ['lightsalmon', 'white']),
                   align = ['center']))

    trace1 = go.Bar(x=dismissals['bowler'].head(10),
                    y=dismissals["points"].head(10),
                    xaxis='x1',
                    yaxis='y1',
                    marker=dict(color='lightblue'),opacity=0.60)

    layout = dict(
        width=830,
        height=410,
        autosize=False,
        title='Bowlers with maximum dismissal points',
        showlegend=False,
        xaxis1=dict(**dict(domain=[0.58, 1], anchor='y1', showticklabels=True)),
        yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),
    )

    fig1 = dict(data=[trace, trace1], layout=layout)
    st.plotly_chart(fig1,use_container_width=True)

    st.markdown("Best Bowlers in each economy range")

    trace1 = go.Scatter(x=e1['bowler'],y = e1['eco_below4'],name='below 4')
    trace2 = go.Scatter(x=e2['bowler'],y = e2['eco_4-5'],name='between 4-5')
    trace3 = go.Scatter(x=e3['bowler'],y = e3['eco_5-6'],name='between 5-6')
    trace4 = go.Scatter(x=e4['bowler'],y = e4['eco_6-9'],name='between 6-9')
    trace5 = go.Scatter(x=e5['bowler'],y = e5['eco_9-10'],name='between 9-10')
    trace6 = go.Scatter(x=e6['bowler'],y = e6['eco_10-11'],name='between 10-11')
    trace7 = go.Scatter(x=e7['bowler'],y = e7['eco_above11'],name='above 11')
    trace8 = go.Scatter(x=m['bowler'],y = m['maiden_overs'],name='Maiden overs')

    fig = tools.make_subplots(rows=4, cols=2,print_grid=False,
                              subplot_titles=('Economy below 4','Economy between 4-5','Economy between 5-6',
                                              'Economy between 6-9','Economy between 9-10','Economy between 10-11',
                                              'Economy above 11','Maiden Overs'))

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 2, 1)
    fig.append_trace(trace4, 2, 2)
    fig.append_trace(trace5, 3, 1)
    fig.append_trace(trace6, 3, 2)
    fig.append_trace(trace7, 4, 1)
    fig.append_trace(trace8, 4, 2)


    fig['layout'].update(height=950, width=850,title='Economy and maiden Overs analysis',showlegend=False)
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("Bowlers with maximum economy and maiden points")

    trace = go.Table(
        domain=dict(x=[0, 0.52],
                    y=[0, 1.0]),
        header=dict(values=["Bowler","Maiden Overs","Points"],
                    fill = dict(color = 'green'),
                    font = dict(color = 'white', size = 14),
                    align = ['center'],
                   height = 30),
        cells=dict(values=[eco['bowler'].head(10),eco['maiden_overs'].head(10),eco['points'].head(10)],
                   fill = dict(color = ['lightgreen', 'white']),
                   align = ['center']))

    trace1 = go.Bar(x=eco['bowler'].head(10),
                    y=eco["points"].head(10),
                    xaxis='x1',
                    yaxis='y1',
                    marker=dict(color='gray'),opacity=0.60,name='bowler')

    layout = dict(
        width=830,
        height=410,
        autosize=False,
        title='Bowlers with maximum economy and maiden points',
        showlegend=False,
        xaxis1=dict(**dict(domain=[0.56, 1], anchor='y1', showticklabels=True)),
        yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),
    )

    fig1 = dict(data=[trace, trace1], layout=layout)
    st.plotly_chart(fig1,use_container_width=True)

    st.markdown("Bowlers with maximum points analysis")


    trace = go.Scatter(y = final_bowl['points'],x = final_bowl['bowler'],mode='markers',
                       marker=dict(size= final_bowl['dismissals'].values,
                                   color = final_bowl['maiden_overs'].values,
                                   colorscale='Viridis',
                                   showscale=True,
                                   colorbar = dict(title = 'Economy')),
                       text = final_bowl['dismissals'].values)

    data = [(trace)]

    layout= go.Layout(autosize= True,
                      title= 'Top Bowlers with maximum points',
                      hovermode= 'closest',
                      xaxis=dict(showgrid=False,zeroline=False,
                                 showline=False),
                      yaxis=dict(title= 'Best Bowlers',ticklen= 5,
                                 gridwidth= 2,showgrid=False,
                                 zeroline=False,showline=False),
                      showlegend= False)

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("Comparison between Top Bowlers")

    data = [go.Scatterpolar(
      r = [x_bowl['points_x'].values[0],x_bowl['points_y'].values[0],x_bowl['points_z'].values[0],x_bowl['points'].values[0]],
      theta = ['Economy rating','Wicket Haul rating','Wickets rating','Total rating'],
      fill = 'toself', opacity = 0.8,
      name = "Harbhajan Singh"),

        go.Scatterpolar(
      r = [y_bowl['points_x'].values[0],y_bowl['points_y'].values[0],y_bowl['points_z'].values[0],y_bowl['points'].values[0]],
      theta = ['Economy rating','Wicket Haul rating','Wickets rating','Total rating'],
      fill = 'toself',subplot = "polar2",
        name = "SP Narine"),

        go.Scatterpolar(
      r = [z_bowl['points_x'].values[0],z_bowl['points_y'].values[0],z_bowl['points_z'].values[0],z_bowl['points'].values[0]],
      theta = ['Economy rating','Wicket Haul rating','Wickets rating','Total rating'],
      fill = 'toself',subplot = "polar3",
        name = "R Ashwin"),

        go.Scatterpolar(
      r = [w_bowl['points_x'].values[0],w_bowl['points_y'].values[0],w_bowl['points_z'].values[0],w_bowl['points'].values[0]],
      theta = ['Economy rating','Wicket Haul rating','Wickets rating','Total rating'],
      fill = 'toself',subplot = "polar4",
        name = "B Kumar")]

    layout = go.Layout(title = "Comparison Between Harbhajan Singh, SP Narine, R Ashwin, B Kumar",

                       polar = dict(radialaxis = dict(visible = True,range = [0, 100]),
                       domain = dict(x = [0, 0.40],y = [0, 0.40])),

                       polar2 = dict(radialaxis = dict(visible = True,range = [0, 100]),
                       domain = dict(x = [0.60, 1],y = [0, 0.40])),

                       polar3 = dict(radialaxis = dict(visible = True,range = [0, 100]),
                       domain = dict(x = [0, 0.40],y = [0.60, 1])),

                       polar4 = dict(radialaxis = dict(visible = True,range = [0, 100]),
                       domain = dict(x = [0.60, 1.0],y = [0.60, 1])))

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,use_container_width=True)


if(fantasy or fantasy_main):
    st.title("Best Players for fantasy team")
    best_batsman = best_batsman.rename(columns={"batsman": "player"})
    best_bowler = best_bowler.rename(columns={"bowler": "player"})
    best_fielder = best_fielder.rename(columns={"fielder": "player"})
    best_player = pd.merge(pd.merge(best_batsman,best_bowler,on='player',how='outer'),best_fielder,on='player',how='outer')

    best_player = best_player.fillna(0)
    best_player['points'] = best_player['total_points']+best_player['points_x']+best_player['points_y']
    best_player.sort_values(by='points',ascending=False,inplace=True)
    best_player=best_player.reset_index().drop(['index'],axis=1)

    best_player = best_player.head(20)
    trace1 = go.Bar(
        x=best_player['player'],
        y=best_player['total_points'],
        name='Batting points',opacity=0.8,
        marker=dict(color='lightblue'))

    trace2 = go.Bar(
        x=best_player['player'],
        y=best_player['points_x'],
        name='Bowling points',opacity=0.7,
        marker=dict(color='gold'))

    trace3 = go.Bar(
        x=best_player['player'],
        y=best_player['points_y'],
        name='Fielding points',opacity=0.7,
        marker=dict(color='lightgreen'))


    data = [trace1, trace2, trace3]
    layout = go.Layout(title="Points Distribution of Top Players",barmode='stack',xaxis = dict(tickmode='linear'),
                                        yaxis = dict(title= "Points Distribution"))

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,use_container_width=True)




x=['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
    'Rising Pune Supergiant', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',
    'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
    'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants', 'Delhi Capitals']

y = ['SRH','MI','GL','RPS','RCB','KKR','DC','KXIP','CSK','RR','SRH','KTK','PW','RPS','DC']

matches.replace(x,y,inplace = True)
deliveries.replace(x,y,inplace = True)

matches_played=pd.concat([matches['team1'],matches['team2']])
matches_played=matches_played.value_counts().reset_index()
matches_played.columns=['Team','Total Matches']
matches_played['wins']=matches['winner'].value_counts().reset_index()['winner']

matches_played.set_index('Team',inplace=True)
matches_played.reset_index().head(8)
win_percentage = round(matches_played['wins']/matches_played['Total Matches'],3)*100

venue_matches=matches.groupby('venue').count()[['id']].sort_values(by='id',ascending=False).head()
ser = pd.Series(venue_matches['id'])

ump=pd.concat([matches['umpire1'],matches['umpire2']])
ump=ump.value_counts()
umps=ump.to_frame().reset_index()

if(team or team_main):
    st.title("Team Analysis")
    st.title("Win percentages of each team")
    trace1 = go.Bar(x=matches_played.index,y=matches_played['Total Matches'],
                    name='Total Matches',opacity=0.4)

    trace2 = go.Bar(x=matches_played.index,y=matches_played['wins'],
                    name='Matches Won',marker=dict(color='red'),opacity=0.4)

    trace3 = go.Bar(x=matches_played.index,
                   y=(round(matches_played['wins']/matches_played['Total Matches'],3)*100),
                   name='Win Percentage',opacity=0.6,marker=dict(color='gold'))

    data = [trace1, trace2, trace3]

    layout = go.Layout(title='Match Played, Wins And Win Percentage',xaxis=dict(title='Team'),
                       yaxis=dict(title='Count'),bargap=0.2,bargroupgap=0.1)

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("Venue of Most matches")
    venue_matches=matches.groupby('venue').count()[['id']].reset_index()

    data = [{"x": venue_matches['id'],"y": venue_matches['venue'],
              "marker": {"color": "lightblue", "size": 12},
             "line": {"color": "red","width" : 2,"dash" : 'dash'},
              "mode": "markers+lines", "name": "Women", "type": "scatter"}]

    layout = {"title": "Stadiums and Matches",
              "xaxis": {"title": "Matches Played", },
              "yaxis": {"title": "Stadiums"},
              "autosize":False,"width":900,"height":1000,
              "margin": go.layout.Margin(l=340, r=0,b=100,t=100,pad=0)}

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,use_container_width = True)

    st.markdown("Favorite Umpire")
    data = [go.Bar(x=umps['index'],y=umps[0],opacity=0.4)]

    layout = go.Layout(title='Umpires in Matches',
                       yaxis=dict(title='Matches'),bargap=0.2)

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,use_container_width = True)

batsmen = matches[['id','season']].merge(deliveries, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)
season=batsmen.groupby(['season'])['total_runs'].sum().reset_index()

avgruns_each_season=matches.groupby(['season']).count().id.reset_index()
avgruns_each_season.rename(columns={'id':'matches'},inplace=1)
avgruns_each_season['total_runs']=season['total_runs']
avgruns_each_season['average_runs_per_match']=avgruns_each_season['total_runs']/avgruns_each_season['matches']
avgruns_each_season.sort_values(by='total_runs', ascending=False)

Season_boundaries=batsmen.groupby("season")["batsman_runs"].agg(lambda x: (x==6).sum()).reset_index()
fours=batsmen.groupby("season")["batsman_runs"].agg(lambda x: (x==4).sum()).reset_index()
Season_boundaries=Season_boundaries.merge(fours,left_on='season',right_on='season',how='left')
Season_boundaries=Season_boundaries.rename(columns={'batsman_runs_x':'6"s','batsman_runs_y':'4"s'})

Season_boundaries['6"s'] = Season_boundaries['6"s']*6
Season_boundaries['4"s'] = Season_boundaries['4"s']*4
Season_boundaries['total_runs'] = season['total_runs']

high_scores=deliveries.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index()
high_scores=high_scores[high_scores['total_runs']>=200]
high_scores.nlargest(10,'total_runs')

high_scores=high_scores[high_scores.batting_team != 'GL']
high_scores=high_scores[high_scores.bowling_team != 'RPS']
high_scores=high_scores[high_scores.bowling_team != 'GL']
high_scores=high_scores[high_scores.bowling_team != 'PW']

high_scores=high_scores.groupby(['inning','batting_team']).count().reset_index()
high_scores.drop(["bowling_team","total_runs"],axis=1,inplace=True)
high_scores.rename(columns={"match_id":"total_times"},inplace=True)

high_scores_1 = high_scores[high_scores['inning']==1]
high_scores_2 = high_scores[high_scores['inning']==2]

high_scores_1.sort_values(by = 'total_times',ascending=False)

high_scores=deliveries.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index()
high_scores1=high_scores[high_scores['inning']==1]
high_scores2=high_scores[high_scores['inning']==2]
high_scores1=high_scores1.merge(high_scores2[['match_id','inning', 'total_runs']], on='match_id')
high_scores1.rename(columns={'inning_x':'inning_1','inning_y':'inning_2','total_runs_x':'inning1_runs','total_runs_y':'inning2_runs'},inplace=True)
high_scores1=high_scores1[high_scores1['inning1_runs']>=200]
high_scores1['is_score_chased']=1
high_scores1['is_score_chased'] = np.where(high_scores1['inning1_runs']<=high_scores1['inning2_runs'], 'yes', 'no')

slices=high_scores1['is_score_chased'].value_counts().reset_index().is_score_chased
list(slices)
labels=['No','Yes']


if(runs_val or runs_val_main):
    fig = {"data" : [{"x" : season["season"],"y" : season["total_runs"],
                      "name" : "Total Run","marker" : {"color" : "lightblue","size": 12},
                      "line": {"width" : 3},"type" : "scatter","mode" : "lines+markers" },

                     {"x" : season["season"],"y" : avgruns_each_season["average_runs_per_match"],
                      "name" : "Average Run","marker" : {"color" : "brown","size": 12},
                      "type" : "scatter","line": {"width" : 3},"mode" : "lines+markers",
                      "xaxis" : "x2","yaxis" : "y2",}],

            "layout" : {"title": "Total and Average run per Season",
                        "xaxis2" : {"domain" : [0, 1],"anchor" : "y2",
                        "showticklabels" : False},"margin" : {"b" : 111},
                        "yaxis2" : {"domain" : [.55, 1],"anchor" : "x2","title": "Average Run"},
                        "xaxis" : {"domain" : [0, 1],"tickmode":'linear',"title": "Year"},
                        "yaxis" : {"domain" :[0, .45], "title": "Total Run"}}}
    st.title("Average and Total Runs analysis")
    st.plotly_chart(fig,use_container_width=True)
    trace1 = go.Bar(
        x=Season_boundaries['season'],
        y=Season_boundaries['total_runs']-(Season_boundaries['6"s']+Season_boundaries['4"s']),
        name='Remaining runs',opacity=0.6)

    trace2 = go.Bar(
        x=Season_boundaries['season'],
        y=Season_boundaries['4"s'],
        name='Run by 4"s',opacity=0.7)

    trace3 = go.Bar(
        x=Season_boundaries['season'],
        y=Season_boundaries['6"s'],
        name='Run by 6"s',opacity=0.7)


    data = [trace1, trace2, trace3]
    layout = go.Layout(title="Run Distribution per year",barmode='stack',xaxis = dict(tickmode='linear',title="Year"),
                                        yaxis = dict(title= "Run Distribution"))

    fig = go.Figure(data=data, layout=layout)
    st.title("Run distribution over years")
    st.plotly_chart(fig,use_container_width=True)
    trace1 = go.Bar(x=high_scores_1['batting_team'],y=high_scores_1['total_times'],name='Ist Innings')
    trace2 = go.Bar(x=high_scores_2['batting_team'],y=high_scores_2['total_times'],name='IInd Innings')

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('At Ist Innings','At IInd Innings'))

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)

    st.title("Number of times teams scored > 200")
    st.plotly_chart(fig,use_container_width=True)
    trace0 = go.Pie(labels=labels, values=slices,
                  hoverinfo='label+value')

    layout=go.Layout(title='percentage of 200 score chased ')
    fig = go.Figure(data=[trace0], layout=layout)
    st.title("Percentage of scores >200 chased")
    st.plotly_chart(fig,use_container_width=True)


agg = matches[['id','season', 'winner', 'toss_winner', 'toss_decision', 'team1']].merge(deliveries, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)
batsman_grp = agg.groupby(["season","match_id", "inning", "batting_team", "batsman"])
batsmen = batsman_grp["batsman_runs"].sum().reset_index()
runs_scored = batsmen.groupby(['season','batting_team', 'batsman'])['batsman_runs'].agg(['sum','mean']).reset_index()
runs_scored['mean']=round(runs_scored['mean'])

agg_battingteam = agg.groupby(['season','match_id', 'inning', 'batting_team', 'bowling_team','winner'])['total_runs'].sum().reset_index()
winner = agg_battingteam[agg_battingteam['batting_team'] == agg_battingteam['winner']]#agg_batting = agg_battingteam.groupby(['season', 'inning', 'team1','winner'])['total_runs'].sum().reset_index()
winner_batting_first = winner[winner['inning'] == 1]
winner_batting_second = winner[winner['inning'] == 2]

winner_runs_batting_first = winner_batting_first.groupby(['season', 'winner'])['total_runs'].mean().reset_index().round()
winner_runs_batting_second = winner_batting_second.groupby(['season', 'winner'])['total_runs'].mean().reset_index().round()

winner_runs = winner_runs_batting_first.merge(winner_runs_batting_second, on = ['season','winner'])
winner_runs.columns = ['season', 'winner', 'batting_first', 'batting_second']

total_win=matches.groupby(['season','winner']).count()[['id']].reset_index()
winner_runs["wins"]= total_win['id']

winner_runs.sort_values(by = ['season'],inplace=True)

csk= winner_runs[winner_runs['winner'] == 'CSK']
rr= winner_runs[winner_runs['winner'] == 'RR']
srh= winner_runs[winner_runs['winner'] == 'SRH']
kkr= winner_runs[winner_runs['winner'] == 'KKR']
mi= winner_runs[winner_runs['winner'] == 'MI']
rcb= winner_runs[winner_runs['winner'] == 'RCB']
kxip= winner_runs[winner_runs['winner'] == 'KXIP']
dd= winner_runs[winner_runs['winner'] == 'DC']
runs_per_over = deliveries.pivot_table(index=['over'],columns='batting_team',values='total_runs',aggfunc=sum)
runs_per_over.reset_index(inplace=True)
runs_per_over.drop(['KTK','PW','RPS','GL'],axis=1,inplace=True)

if(avg or avg_main):
    trace1 = go.Scatter(x=csk['season'],y = csk['batting_first'],name='Batting First')
    trace2 = go.Scatter(x=csk['season'],y = csk['batting_second'],name='Batting Second')
    trace3 = go.Scatter(x=rr['season'],y = rr['batting_first'],name='Batting First')
    trace4 = go.Scatter(x=rr['season'],y = rr['batting_second'],name='Batting Second')
    trace5 = go.Scatter(x=srh['season'],y = srh['batting_first'],name='Batting First')
    trace6 = go.Scatter(x=srh['season'],y = srh['batting_second'],name='Batting Second')
    trace7 = go.Scatter(x=kkr['season'],y = kkr['batting_first'],name='Batting First')
    trace8 = go.Scatter(x=kkr['season'],y = kkr['batting_second'],name='Batting Second')
    trace9 = go.Scatter(x=rcb['season'],y = rcb['batting_first'],name='Batting First')
    trace10 = go.Scatter(x=rcb['season'],y = rcb['batting_second'],name='Batting Second')
    trace11 = go.Scatter(x=kxip['season'],y = kxip['batting_first'],name='Batting First')
    trace12 = go.Scatter(x=kxip['season'],y = kxip['batting_second'],name='Batting Second')
    trace13 = go.Scatter(x=mi['season'],y = mi['batting_first'],name='Batting First')
    trace14 = go.Scatter(x=mi['season'],y = mi['batting_second'],name='Batting Second')
    trace15 = go.Scatter(x=dd['season'],y = dd['batting_first'],name='Batting First')
    trace16 = go.Scatter(x=dd['season'],y = dd['batting_second'],name='Batting Second')

    fig = tools.make_subplots(rows=4, cols=2, subplot_titles=('CSK', 'RR','SRH', 'KKR','RCB', 'KXIP','MI', 'DC'))

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 1, 2)
    fig.append_trace(trace4, 1, 2)
    fig.append_trace(trace5, 2, 1)
    fig.append_trace(trace6, 2, 1)
    fig.append_trace(trace7, 2, 2)
    fig.append_trace(trace8, 2, 2)
    fig.append_trace(trace9, 3, 1)
    fig.append_trace(trace10, 3, 1)
    fig.append_trace(trace11, 3, 2)
    fig.append_trace(trace12, 3, 2)
    fig.append_trace(trace13, 4, 1)
    fig.append_trace(trace14, 4, 1)
    fig.append_trace(trace15, 4, 2)
    fig.append_trace(trace16, 4, 2)

    fig['layout'].update(title='Batting first vs Batting Second of Teams',showlegend=False)
    st.plotly_chart(fig,use_container_width=True)

    trace1 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['CSK'],name='CSK',marker= dict(color= "blue",size=12))
    trace2 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['DC'],name='DC')
    trace3 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['KKR'],name='KKR')
    trace4 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['KXIP'],name='KXIP')
    trace5 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['MI'],name='MI')
    trace6 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['RCB'],name='RCB')
    trace7 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['RR'],name='RR')
    trace8 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['SRH'],name='SRH')

    data = [trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8]

    layout = go.Layout(title='Average Run in Each Over',xaxis = dict(tickmode='linear',title="Over"),
                                        yaxis = dict(title= "Runs"))

    fig = go.Figure(data=data,layout=layout)
    st.plotly_chart(fig,use_container_width=True)

season=matches[['id','season','winner']]
complete_data=deliveries.merge(season,how='inner',left_on='match_id',right_on='id')

powerplay_data=complete_data[complete_data['over']<=6]

inn1 = powerplay_data[ powerplay_data['inning']==1].groupby('match_id')['total_runs'].agg(['sum']).reset_index()
inn2 = powerplay_data[ powerplay_data['inning']==2].groupby('match_id')['total_runs'].agg(['sum']).reset_index()

inn1.reset_index(inplace=True)
inn1.drop(["match_id"],axis=1,inplace=True)

inn2.reset_index(inplace=True)
inn2.drop(["match_id"],axis=1,inplace=True)

pi1=powerplay_data[ powerplay_data['inning']==1].groupby(['season','match_id'])['total_runs'].agg(['sum'])
pi1=pi1.reset_index().groupby('season')['sum'].mean()
pi1=pi1.to_frame().reset_index()

pi2=powerplay_data[ powerplay_data['inning']==2].groupby(['season','match_id'])['total_runs'].agg(['sum'])
pi2=pi2.reset_index().groupby('season')['sum'].mean()
pi2=pi2.to_frame().reset_index()

powerplay_dismissals=powerplay_data.dropna(subset=['dismissal_kind']).groupby(['season','match_id','inning'])['dismissal_kind'].agg(['count']).reset_index().groupby('season')['count'].max()
powerplay_dismissals=powerplay_dismissals.reset_index()

powerplay_dismissals_first=powerplay_data[ powerplay_data['inning']==1].dropna(subset=['dismissal_kind']).groupby(['season','match_id','inning'])['dismissal_kind'].agg(['count']).reset_index().groupby('season')['count'].mean()
powerplay_dismissals_first=powerplay_dismissals_first.reset_index()

powerplay_dismissals_second=powerplay_data[ powerplay_data['inning']==2].dropna(subset=['dismissal_kind']).groupby(['season','match_id','inning'])['dismissal_kind'].agg(['count']).reset_index().groupby('season')['count'].mean()
powerplay_dismissals_second=powerplay_dismissals_second.reset_index()

if(powerplay or powerplay_main):
    st.title("Powerplay Analysis")
    fig = {"data" : [{"x" : inn1["index"],"y" : inn1["sum"],"marker" : {"color" : "blue","size": 2},
                      "line": {"width" : 1.5},"type" : "scatter","mode" : "lines" },

                     {"x" : inn2["index"],"y" : inn2["sum"],"marker" : {"color" : "brown","size": 2},
                      "type" : "scatter","line": {"width" : 1.5},"mode" : "lines",
                      "xaxis" : "x2","yaxis" : "y2",}],

            "layout" : {"title": "Inning 1 vs Inning 2 in Powerplay Overs",
                        "xaxis2" : {"domain" : [0, 1],"anchor" : "y2",
                        "showticklabels" : False},
                        "yaxis2" : {"domain" : [.55, 1],"anchor" : "x2","title": "Inn2 Powerplay"},
                        "margin" : {"b" : 111},
                        "xaxis" : {"domain" : [0, 1],"title": "Matches"},
                        "yaxis" : {"domain" :[0, .45], "title": "Inn1 Poweplay"}}}

    st.plotly_chart(fig,use_container_width=True)

    trace1 = go.Bar(x=pi1.season,y=pi1["sum"],
                    name='Inning 1',opacity=0.4)

    trace2 = go.Bar(x=pi2.season,y=pi2["sum"],name='Inning 2',
                    marker=dict(color='red'),opacity=0.4)

    data = [trace1, trace2]
    layout = go.Layout(title='Powerplay Average runs per Year',
                       xaxis=dict(title='Year',tickmode='linear'),
                       yaxis=dict(title='Run'),bargap=0.2,bargroupgap=0.1)

    fig = go.Figure(data=data, layout=layout)
    st.markdown("Powerplay Average Runs")
    st.plotly_chart(fig,use_container_width=True)
    trace1 = go.Bar(x=powerplay_dismissals.season,y=powerplay_dismissals["count"],
                    name='Max',opacity=0.4)

    trace2 = go.Bar(x=powerplay_dismissals_first.season,y=powerplay_dismissals_first["count"],name='Inning 1',
                    marker=dict(color='red'),opacity=0.4)

    trace3 = go.Bar(x=powerplay_dismissals_second.season,y=powerplay_dismissals_second["count"],name='Inning 2',
                    marker=dict(color='lime'),opacity=0.4)

    data = [trace1, trace2, trace3]
    layout = go.Layout(title='Powerplay Average Dismissals per Year',
                       xaxis=dict(title='Year',tickmode='linear'),
                       yaxis=dict(title='Run'),bargap=0.2,bargroupgap=0.1)

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig,use_container_width=True)
