## Inviting our friends to the party
import pandas as pd
import networkx as nx
from datetime import datetime as dt, timedelta as td
from docplex.mp.model import Model

## Configs and Parameters
# Driver Schedule Input File
schedule_file = "./data/RouteDetails.xlsx"
preload_time = 6
drop_time = 8
loading_type = 'DAHK'


# Function to read in optimized driver schedule as DataFrame
def read_driver_schedule(file_loc, sample_size=None):
    """Function for pre-processing input."""
    cols = ['Route', 'Sequence', 'Zip', 'EqCode', 'City', 
            'State', 'ArvDate', 'ArvTime', 'DeptDate', 'DeptTime']
    
    df = (pd.read_excel(file_loc, usecols=cols)
          .sort_values(by=['Route', 'Sequence'])
          .assign(PreloadTm=preload_time, DropTm=drop_time, LoadAct=loading_type,
                  ArvDTTM=lambda x: pd.to_datetime(x.ArvDate+x.ArvTime, format="%Y-%m-%d%H:%M:%S"),
                  DeptDTTM=lambda x: pd.to_datetime(x.DeptDate+x.DeptTime, format="%Y-%m-%d%H:%M:%S"))
          .drop(columns=['ArvDate', 'ArvTime', 'DeptDate', 'DeptTime']))
    
    df['NxtArvDTTM'] = df.ArvDTTM.shift(-1)
    df['DestZip'] = df.Zip.shift(-1)
    
    # drop last sequence per route to form legs
    df = df[df.groupby(['Route'])['Sequence'].transform(max) != df['Sequence']]
    if sample_size:
        return df.query("Route<=@sample_size")
    else: return df
    
    
def build_network_graph(schedule_file):
    drv_sched = read_driver_schedule(schedule_file)
    
    #-- pseudo-vars --
    locations = drv_sched.Zip.unique()
    routes = drv_sched.Route.unique()
    
    #-- instantiate directed graph
    G = nx.DiGraph()
    
    ##-- create nodes
    # source and target nodes
    G.add_node('s', tag='source')
    G.add_node('t', tag='target')
    
    # location nodes (Zip used to identify unique locs)
    for loc in locations:
        for typ in ('out', 'in'):
            G.add_node((loc, typ), tag='location')

    # trailer utilization nodes
    drv_sched.apply(lambda x: G.add_node((x.Route, x.Sequence, 'start'),
                                         start_loc=x.Zip,
                                         start=x.ArvDTTM-td(hours=preload_time),
                                         tag='trailer_start'), axis=1)
    
    drv_sched.apply(lambda x: G.add_node((x.Route, x.Sequence, 'end'),
                                         end_loc=x.DestZip,
                                         end=x.NxtArvDTTM+td(hours=drop_time),
                                         tag='trailer_end'), axis=1)
    
    # source - pool edges
    G.add_edges_from([('s', (loc, 'out')) for loc in locations], tag='src_pool')
    
    # pool - sink edges
    G.add_edges_from([((loc, 'in'), 't') for loc in locations], tag='sink_pool')
    
    # trailer supplied from pool edges
    drv_sched.apply(lambda x: G.add_edge((x.Zip, 'out'), (x.Route, x.Sequence, 'start'), tag='trl_supp'), axis=1)
    
    # trailer returned to pool edges
    drv_sched.apply(lambda x: G.add_edge((x.Route, x.Sequence, 'end'), (x.DestZip, 'in'), tag='trl_rtrn'), axis=1)
    
    # trailer utilization edges
    trailer_start = [x for x,y in G.nodes(data=True) if y['tag']=='trailer_start']
    trailer_end = [x for x,y in G.nodes(data=True) if y['tag']=='trailer_end']
    G.add_edges_from([(i, j) for i in trailer_start
                                for j in trailer_end
                                    if i[:2]==j[:2]], tag='trailer_utilization')    
        
    # same trailer used edges
    # same if end time is less then next start time, to location is same as next from loc
    G.add_edges_from([(i, j) for i in trailer_end 
                                for j in trailer_start
                                    if G.node[i]['end_loc']==G.node[j]['start_loc'] 
                                      and G.node[i]['end']<G.node[j]['start']], tag='same_trailer')
    
    return G   


def solve_min_flow_prob(schedule_file, bal_pool=False):
    G = build_network_graph(schedule_file)
    # instantiate model
    mdl = Model(name='Trailer Planning Model')
    
    # create flow variables
    flow_var = {}
    for i, j in G.edges():
        if G.edges[i, j]['tag'] in ('src_pool', 'sink_pool', 'trl_supp', 'trl_rtrn', 'same_trailer'):
            flow_var[i, j]=mdl.continuous_var(lb=0, name=f"{i}_{j}_{G.edges[i, j]['tag']}")
        else:
            flow_var[i, j]=mdl.continuous_var(lb=int(1), ub=int(1), name=f"{i}_{j}_{G.edges[i, j]['tag']}")
                                              
    # flow value variable for total trailers   
    v = mdl.continuous_var(name='Flow value')
    
    # add flow conservation constraints
    for i in G.nodes():
        if i=='s':
            mdl.add_constraint(mdl.sum_vars(flow_var[i, j] for j in G.neighbors(i))
                              == v)
        elif i=='t':
            mdl.add_constraint(mdl.sum_vars(flow_var[m, n] for m, n in G.edges() if n==i)
                              == v)
        else:
            mdl.add_constraint(mdl.sum_vars(flow_var[i, j] for j in G.neighbors(i))
                              - mdl.sum_vars(flow_var[m, n] for m, n in G.edges() if n==i)
                              == 0)
            # pool balance constraints
            if  bal_pool and G.node[i]['tag']=='location' and i[1]=='out':
                mdl.add_constraint(mdl.sum(flow_var['s', i])
                                  - mdl.sum(flow_var[(i[0], 'in'), 't'])
                                  == 0)

    mdl.minimize(v)
    mdl.print_information()
    mdl.solve(log_output=True)
    if not mdl.solve():
        print('*** Problem has no solution')
    else:
        sol = {}
        obj = v.solution_value
        print(f"Minimum trailers required: {obj}")
        for i, j in G.edges():
            sol[i, j] = flow_var[i, j].solution_value
        G.clear()
        return sol
                                              

if __name__ == __main__:
    solve_min_flow_prob(schedule_file, balance_pool=True)