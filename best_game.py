import numpy as np
import pandas as pd

#define the precision of the grid used to solve the problem
grid_size = 20
grid = np.array([float(i)/grid_size for i in range(grid_size+1)])
precision = 1./(3*grid_size)


#define a type to use in the numpy arrays that handle probability and game values (the number that each player played)
game_type = [('player', int), ('value', float)]

def calculate_prob(values):
    'Get the probability of each number to win the game given its position'
    
    #sort the values because it is easier to calculate the probability if they are sorted
    values_sorted = np.sort(values,order='value')
    diff_values = np.diff(values_sorted['value']).T
    values_sorted = values_sorted.T
    
    #the first and last values are calculated separately and they are all joined into a single array
    p0 = values_sorted['value'][0] + diff_values[0]/2.
    pn = (diff_values[:-1] + diff_values[1:])/2.
    p_last = 1.0 - values_sorted['value'][-1] + diff_values[-1]/2.
    prob_values = np.concatenate([[p0],pn,[p_last]]).T
    
    #move it back to the format of 'game_type'
    probability_array = np.zeros(values.T.shape, dtype=game_type)
    probability_array['player'] = values_sorted['player']
    probability_array['value'] = prob_values.T
    prob = probability_array.T
    
    #sort values to have again player 0...n 
    prob.sort(order='player')
    
    return prob


def get_values_to_search(current_variable,values_variables,variables,grid_solve):
    'Get the allowed values of the variables, given how each player play'
    if (current_variable < variables[-1]):
        variables_already_solved = variables[variables>current_variable]
        next_variable = variables_already_solved[0] #get the next variable
        
        #for each value of the current variable, finds the possible values of the play
        #the next players are called recursively to get their best plays
        values_to_search = list()
        
        for current_play in grid_solve:
            
            values_variables_next_player = np.append(values_variables,np.array([(current_variable,current_play)],dtype=game_type))            
            #use as input the current value in the loop
            best_plays_for_next_players = get_best_play(next_variable,
                                                        values_variables_next_player,
                                                        variables,
                                                        grid_solve)
            for x in best_plays_for_next_players:
                already_solved_values = np.empty( len(variables_already_solved),dtype=game_type )
                already_solved_values['value'] = x
                already_solved_values['player'] = variables_already_solved
                play =  np.append(values_variables,already_solved_values)
                play = np.append(play,np.array([(current_variable,current_play)],dtype=game_type))
                
                values_to_search.append(play)
            
    else: #last variable: it should start from the grid values
        values_to_search = [np.append(values_variables,np.array((current_variable,x),dtype=game_type)) for x in grid_solve]
    
    values_to_search = np.array(values_to_search,dtype=game_type)
    #order the values to have player 0 ... n again
    values_to_search.sort(order='player')
    
    return values_to_search


def get_maximum_probability(current_variable,prob):
    'Returns the values of the current variable with maximum probability'
    #In case the next players have more than one possibility of play, they will play randomly
    #So each of those plays has equal probabilities and the mean of the probabilities for current variable is taken
    best_mean_prob = prob.groupby(by=str(current_variable),sort=False).mean()

    prob_max = np.abs((best_mean_prob["prob_"+str(current_variable)]-best_mean_prob["prob_"+str(current_variable)].max())) <= precision
    best_mean_prob = best_mean_prob[prob_max]
    
    best_values = best_mean_prob.index
    return best_values


def get_best_play(current_variable, values_variables, variables=np.array(list('012')),grid_values=grid):
    'Given the other values of the variables, calculate the best play'
    
    grid_solve = grid_values.copy()
    next_variables = np.array(variables[variables>=current_variable],dtype=str)
    
    #the numbers cannot be the same, so they are removed from the grid
    if (current_variable > variables[0]):
        for variable in values_variables['value']:
            grid_solve = grid_solve[np.where(np.abs(grid_solve-variable)>precision)]
    else: #no need to remove the values for the first variable, since it has no previous values 
        values_variables = np.array([],dtype=game_type)
    
    #define the value that the variables will take during the probability calculation
    values_to_search = get_values_to_search(current_variable,values_variables,variables,grid_solve)
                
    #calculate the probability
    prob = calculate_prob(values_to_search)
    #convert it into a DataFrame to make it easy to use
    prob = pd.DataFrame(prob['value'],columns=list(map(lambda x: "prob_"+str(x), variables)))
    
    #add the information of the variables played to the DataFrame
    values_played = pd.DataFrame(values_to_search['value'],columns=list(map(str, variables)) ) 
    for x in np.array(variables,dtype=str):
        prob[x] = values_played[x]     
    
    best_values = get_maximum_probability(current_variable,prob)
    
    #The function should return the values all plays that contain 
    prob = prob.loc[prob[str(current_variable)].isin(best_values)]
    
    return np.array(prob[next_variables])

def optimal_values_for_first_player(number_of_players):
    other_plays = np.array([],dtype=game_type) #the first player does not know anything when he/she plays
    a_plays = get_best_play(0,other_plays, np.array(range(number_of_players)))
    return list(np.unique(a_plays.T[0]))


print('The best values that player A can play in case there are 3 players are ' + str(optimal_values_for_first_player(3)))


print('The best values that player A can play in case there are 4 players are ' + str(optimal_values_for_first_player(4)))

