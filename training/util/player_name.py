import pybaseball

def get_player_name(id):
    player_info = pybaseball.playerid_reverse_lookup([id])
    return player_info['name_first'][0] + " " + player_info['name_last'][0]