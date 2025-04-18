from mpd import MPDClient


mpd_client = MPDClient()
mpd_client.connect("localhost", 6600)
