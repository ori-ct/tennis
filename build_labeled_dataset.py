#!env/bin/python3

import urllib.request
import json
import sys
import numpy as np
import os

def progress_bar(iteration, total, barLength=50):
	percent = int(round((iteration / total) * 100))
	nb_bar_fill = int(round((barLength * percent) / 100))
	bar_fill = '#' * nb_bar_fill
	bar_empty = ' ' * (barLength - nb_bar_fill)
	sys.stdout.write("\r  [{0}] {1}%".format(str(bar_fill + bar_empty), percent))
	sys.stdout.flush()

def search_places(location,radius,keyword,APIkey):
	url  = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?'
	url += 'location='+str(location[0])+','+str(location[1])
	url += '&radius='+str(radius)
	url += '&keyword='+keyword
	url += '&key='+APIkey
	results = []
	while len(results)<1:
		with urllib.request.urlopen(url) as response:
			data = response.read()
		data_struct = json.loads(data)
		if 'error_message' in data_struct.keys():
			raise NameError(data_struct['error_message'])
		results.extend(data_struct['results'])
		if 'next_page_token' in data_struct.keys():
			url  = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?'
			url += 'pagetoken='+data_struct['next_page_token']
			url += '&key='+APIkey
		else:
			break
	return results

def download_image(location,zoom,imsize,scale,fname,APIkey):
	url = 'https://maps.googleapis.com/maps/api/staticmap?'
	url += 'center='+str(location[0])+','+str(location[1])
	url += '&zoom='+str(zoom)
	url += '&size='+str(imsize[0])+'x'+str(imsize[1])
	url += '&scale='+str(scale)
	url += '&maptype=satellite&key='+APIkey
	with urllib.request.urlopen(url) as response, open(fname,'wb') as out_file:
		data = response.read()
		out_file.write(data)


if len(sys.argv)==1:
	print('Unexpected number of parameters.')
	print('Help: ' + sys.argv[0] + ' -h')
	exit()
if len(sys.argv)>=2:
	if sys.argv[1]=='-h':
		print('Reuse dataset:')
		print('   ' + sys.argv[0] + ' <google_maps_API_key> <data_destination>')
		print('\nSearch new dataset:')
		print('   ' + sys.argv[0] + ' <google_maps_API_key> <data_destination> <google_places_API_key>')
		print('\nObtain Google Maps   credentials at https://developers.google.com/maps/documentation/javascript/get-api-key')
		print('Obtain Google Places credentials at https://developers.google.com/places/web-service/get-api-key')
		exit()
	else:
		APIkey_maps = sys.argv[1]
		data_destination = sys.argv[2]
if len(sys.argv)>4:
	print('Unexpected number of parameters.')
	print('Help: ' + sys.argv[0] + ' -h')
	exit()
if len(sys.argv)==4:
	print('Are you sure you want to overwrite the places dataset with new data? [Y/n]')
	if input()=='n':
		exit()

	APIkey_places=sys.argv[3]
	radius = 50000
	data = []
	with open('cities.json','rb') as f:
		cities = json.loads(f.read())
	#keywords = ['tennis+court','soccer+field','football+field','softball+field','track+and+field','turf+field']
	keywords = ['soccer+field','track+and+field','turf+field']
	for keyword in keywords:
		for c in cities:
			location =(c['latitude'],c['longitude'])
			try:
				data_search = search_places(location,radius,keyword,APIkey_places)
			except:
				print(sys.exc_info()[1])
				print('Quitting search, saving results...')
				break
			for ind in range(len(data_search)):
				data_search[ind]['city']=c['city']
				data_search[ind]['state']=c['state']
				data_search[ind]['keyword']=keyword
			data.extend(data_search)
			print(c['city']+', '+c['state'] + ': '+str(len(data_search))+' '+keyword+'s - Total: '+str(len(data))+' '+keyword+'s found.')
	with open('data_new.json','w') as outfile:
		json.dump(data, outfile)

else:
	with open('data_places.json' ,'rb') as f:
		data = json.loads(f.read())

print('Downloading dataset from Google Maps...')

zoom =18
imsize = (256,256)
scale = 2
if data_destination[-1]!='/':
	data_destination = data_destination[:-1]

for iteration,place in zip(range(len(data)),data):
	progress_bar(iteration, len(data),30)
	if place['keyword']=='tennis+court':
		fname = data_destination + 'tennis+court/' + place['place_id'] + '.png'
	else:
		fname = data_destnation + 'hard/' + place['place_id'] + '.png'
	if os.path.isfile(fname):
		continue
	place_location = (place['geometry']['location']['lat'],place['geometry']['location']['lng'])
	download_image(place_location,zoom,imsize,scale,fname,APIkey_maps)


