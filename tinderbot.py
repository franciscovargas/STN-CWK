import argparse
from datetime import datetime
import json
from random import randint
import requests
import sys
from time import sleep
import pickle
import re
import multiprocessing
import time
import string
import random


headers = {
    'app_version': '3',
    'platform': 'ios',
}
dictionary4 = {}
dictionary8 = {}
dictionary12 = {}
dictionary16 = {}
dictionary20 = {}
dictionary24 = {}
dictionary28 = {}
dictionary32 = {}
dictionary36 = {}
dictionary40 = {}
dictionary44 = {}
dictionary48 = {}
dictionary52 = {}
dictionary56 = {}
dictionary60 = {}
dictionary64 = {}


fb_id = '<inser facebook id here>'
fb_auth_token = '<insert facebook auth here>' 

class User(object):
    def __init__(self, data_dict):
        self.d = data_dict

    @property
    def user_id(self):
        return self.d['_id']

    @property
    def ago(self):
        raw = self.d.get('ping_time')
        if raw:
            d = datetime.strptime(raw, '%Y-%m-%dT%H:%M:%S.%fZ')
            secs_ago = int(datetime.now().strftime("%s")) - int(d.strftime("%s"))
            if secs_ago > 86400:
                return u'{days} days ago'.format(days=secs_ago / 86400)
            elif secs_ago < 3600:
                return '[unknown]'

    @property
    def bio(self):
        try:
            x = self.d['bio'].encode('ascii', 'ignore').replace('\n', '')[:50].strip()
        except (UnicodeError, UnicodeEncodeError, UnicodeDecodeError):
            return '[garbled]'
        else:
            return x

    @property
    def age(self):
        raw = self.d.get('birth_date')
        if raw:
            d = datetime.strptime(raw, '%Y-%m-%dT%H:%M:%S.%fZ')
            return datetime.now().year - int(d.strftime('%Y'))

        return 0

    def __unicode__(self):
        # print self.d.keys()
        # print self.keys()
        
        #string = []
        #string.append("Name: ")
        #string.append(self.d['name'].encode('utf8'))
        #string.append("Miles: ")
        #string.append(self.d['distance_mi'])
        #string.append("Bio: ")
        tmp = (self.d['bio'].encode('ascii','ignore'))
        tmp = re.sub("[^A-Za-z' ]+", "", tmp)
        name = (''.join(random.choice(string.ascii_uppercase) for i in range(12)))
        #string.append(tmp)
        #text_file.write("Name: {0} Miles: {1} Bio: {2}".format(self.d['name'],self.d['distance_mi'],(self.d['bio'].encode('utf8'))))
        #text_file.write("Person: {0}".format(string))
        dist = self.d['distance_mi']
        if (dist != None):
            
            if(int(dist)<4):
                dictionary4[name] = tmp
                dictionary8[name] = tmp
                dictionary12[name] = tmp
                dictionary16[name] = tmp
                dictionary20[name] = tmp
                dictionary24[name] = tmp
                dictionary28[name] = tmp
                dictionary32[name] = tmp
                dictionary36[name] = tmp
                dictionary40[name] = tmp
                dictionary44[name] = tmp
                dictionary48[name] = tmp
                dictionary52[name] = tmp
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp     
            elif(int(dist)<8):
                dictionary8[name] = tmp
                dictionary12[name] = tmp
                dictionary16[name] = tmp
                dictionary20[name] = tmp
                dictionary24[name] = tmp
                dictionary28[name] = tmp
                dictionary32[name] = tmp
                dictionary36[name] = tmp
                dictionary40[name] = tmp
                dictionary44[name] = tmp
                dictionary48[name] = tmp
                dictionary52[name] = tmp
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp 
            elif(int(dist)<12):
                dictionary12[name] = tmp
                dictionary16[name] = tmp
                dictionary20[name] = tmp
                dictionary24[name] = tmp
                dictionary28[name] = tmp
                dictionary32[name] = tmp
                dictionary36[name] = tmp
                dictionary40[name] = tmp
                dictionary44[name] = tmp
                dictionary48[name] = tmp
                dictionary52[name] = tmp
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp 
            elif(int(dist)<16):
                dictionary16[name] = tmp
                dictionary20[name] = tmp
                dictionary24[name] = tmp
                dictionary28[name] = tmp
                dictionary32[name] = tmp
                dictionary36[name] = tmp
                dictionary40[name] = tmp
                dictionary44[name] = tmp
                dictionary48[name] = tmp
                dictionary52[name] = tmp
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp 
                print "Inside", int(dist) 
            elif(int(dist)<20):
                dictionary20[name] = tmp
                dictionary24[name] = tmp
                dictionary28[name] = tmp
                dictionary32[name] = tmp
                dictionary36[name] = tmp
                dictionary40[name] = tmp
                dictionary44[name] = tmp
                dictionary48[name] = tmp
                dictionary52[name] = tmp
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp 
            elif(int(dist)<24):
                dictionary24[name] = tmp
                dictionary28[name] = tmp
                dictionary32[name] = tmp
                dictionary36[name] = tmp
                dictionary40[name] = tmp
                dictionary44[name] = tmp
                dictionary48[name] = tmp
                dictionary52[name] = tmp
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp 
            elif(int(dist)<28):
                dictionary28[name] = tmp
                dictionary32[name] = tmp
                dictionary36[name] = tmp
                dictionary40[name] = tmp
                dictionary44[name] = tmp
                dictionary48[name] = tmp
                dictionary52[name] = tmp
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp  
            elif(int(dist)<32):
                dictionary32[name] = tmp
                dictionary36[name] = tmp
                dictionary40[name] = tmp
                dictionary44[name] = tmp
                dictionary48[name] = tmp
                dictionary52[name] = tmp
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp 
            elif(int(dist)<36):
                dictionary36[name] = tmp
                dictionary40[name] = tmp
                dictionary44[name] = tmp
                dictionary48[name] = tmp
                dictionary52[name] = tmp
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp
            elif(int(dist)<40):
                dictionary40[name] = tmp
                dictionary44[name] = tmp
                dictionary48[name] = tmp
                dictionary52[name] = tmp
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp 
            elif(int(dist)<44):
                dictionary44[name] = tmp
                dictionary48[name] = tmp
                dictionary52[name] = tmp
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp
            elif(int(dist)<48):
                dictionary48[name] = tmp
                dictionary52[name] = tmp
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp
            elif(int(dist)<52):
                dictionary52[name] = tmp
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp
            elif(int(dist)<56):
                dictionary56[name] = tmp
                dictionary60[name] = tmp
                dictionary64[name] = tmp
            elif(int(dist)<60):
                dictionary60[name] = tmp
                dictionary64[name] = tmp
 
            else:
                dictionary64[name] = tmp 
                print "Inside", int(dist )     
            with open('biodictdist4.pickle', 'wb') as handle:
                pickle.dump(dictionary4, handle)
            with open('biodictdist8.pickle', 'wb') as handle:
                pickle.dump(dictionary8, handle)
            with open('biodictdist12.pickle', 'wb') as handle:
                pickle.dump(dictionary12, handle)
            with open('biodictdist12.pickle', 'wb') as handle:
                pickle.dump(dictionary12, handle)
            with open('biodictdist16.pickle', 'wb') as handle:
                pickle.dump(dictionary16, handle)
            with open('biodictdist20.pickle', 'wb') as handle:
                pickle.dump(dictionary20, handle)
            with open('biodictdist24.pickle', 'wb') as handle:
                pickle.dump(dictionary24, handle)
            with open('biodictdist28.pickle', 'wb') as handle:
                pickle.dump(dictionary28, handle)
            with open('biodictdist32.pickle', 'wb') as handle:
                pickle.dump(dictionary32, handle)
            with open('biodictdist36.pickle', 'wb') as handle:
                pickle.dump(dictionary36, handle)
            with open('biodictdist40.pickle', 'wb') as handle:
                pickle.dump(dictionary40, handle)
            with open('biodictdist44.pickle', 'wb') as handle:
                pickle.dump(dictionary44, handle)
            with open('biodictdist48.pickle', 'wb') as handle:
                pickle.dump(dictionary48, handle)
            with open('biodictdist52.pickle', 'wb') as handle:
                pickle.dump(dictionary52, handle)
            with open('biodictdist56.pickle', 'wb') as handle:
                pickle.dump(dictionary56, handle)
            with open('biodictdist60.pickle', 'wb') as handle:
                pickle.dump(dictionary60, handle)
            with open('biodictdist64.pickle', 'wb') as handle:
                pickle.dump(dictionary64, handle)
        return u'{name} ({age}), {distance}km, {bio1}'.format(
            name=self.d['name'],
            age=self.age,
            distance=dist,
            bio1=(tmp)

        )


def auth_token(fb_auth_token, fb_user_id):
    h = headers
    h.update({'content-type': 'application/json'})
    req = requests.post(
        'https://api.gotinder.com/auth',
        headers=h,
        data=json.dumps({'facebook_token': fb_auth_token, 'facebook_id': fb_user_id})
    )
    try:
        return req.json()['token']
    except:
        return None


def recommendations(auth_token):
    h = headers
    h.update({'X-Auth-Token': auth_token})
    r = requests.get('https://api.gotinder.com/user/recs', headers=h)
    # print r.json().items()[-1][-1]
    # bbb
    if r.status_code == 401 or r.status_code == 504:
        raise Exception('Invalid code')
        print r.content

    if not 'results' in r.json():
        print r.json()

    for result in r.json()['results']:
        yield User(result)


def like(user_id):
    try:
        u = 'https://api.gotinder.com/like/%s' % user_id
        d = requests.get(u, headers=headers, timeout=0.7).json()
    except KeyError:
        raise
    else:
        return d['match']


def nope(user_id):
    try:
        u = 'https://api.gotinder.com/pass/%s' % user_id
        requests.get(u, headers=headers, timeout=0.7).json()
    except KeyError:
        raise


def like_or_nope():
    return 'nope' if randint(1, 100) == 31 else 'like'
    
def mst():
    #text_file = open("bio1.txt", "wb")

    parser = argparse.ArgumentParser(description='Tinder automated bot')
    parser.add_argument('-l', '--log', type=str, default='activity.log', help='Log file destination')

    args = parser.parse_args()

    print 'Tinder bot'
    print '----------'
    matches = 0
    liked = 0
    nopes = 0
    

    while True:
        token = auth_token(fb_auth_token, fb_id)

        if not token:
            print 'could not get token'
            sys.exit(0)

        for user in recommendations(token):
            if not user:
                break
            # print user
            print unicode(user)

            # try:
            #     action = like_or_nope()
            #     if action == 'like':
            #         print ' -> Like'
            #         match = like(user.user_id)
            #         if match:
            #             print ' -> Match!'

            #         with open('./liked.txt', 'a') as f:
            #             f.write(user.user_id + u'\n')

            #     else:
            #         print ' -> random nope :('
            #         nope(user.user_id)

            # except:
            #     print 'networking error %s' % user.user_id

            s = float(randint(250, 2500) / 1000)
        #sleep(10)


def main():
    p = multiprocessing.Process(target=mst, name="tinderbot")
    p.start()

    # Wait 10 seconds for foo
    time.sleep(700)

    # Terminate foo
    #p.terminate()
    # Cleanup
    #p.join()
    print "Reaches this far"
    k = 0

    p.terminate()
    dic = pickle.load( open( "biodictdist4.pickle", "rb" ) )
    for key in dic:
        k = k+1
    print "Dist4", dic
    k=0
    dic = pickle.load( open( "biodictdist8.pickle", "rb" ) )
    for key in dic:
        k = k+1
    print "Dist8", dic
    k=0
    dic = pickle.load( open( "biodictdist16.pickle", "rb" ) )
    for key in dic:
        k = k+1
    print "Dist16", dic
    k=0
    dic = pickle.load( open( "biodictdist32.pickle", "rb" ) )
    for key in dic:
        k = k+1
    print "Dist32", dic 
    k=0
    dic = pickle.load( open( "biodictdist64.pickle", "rb" ) )
    for key in dic:
        k = k+1
    print "Dist64", dic


if __name__ == '__main__':
        # Start foo as a process
    p = multiprocessing.Process(target=mst, name="tinderbot")
    p.start()

    # Wait 10 seconds for foo
    time.sleep(700)

    # Terminate foo
    #p.terminate()
    # Cleanup
    #p.join()
    print "Reaches this far"
    k = 0

    p.terminate()
    dic = pickle.load( open( "biodictdist4.pickle", "rb" ) )
    for key in dic:
        k = k+1
    print "Dist4", k
    k=0
    dic = pickle.load( open( "biodictdist8.pickle", "rb" ) )
    for key in dic:
        k = k+1
    print "Dist8", k
    k=0
    dic = pickle.load( open( "biodictdist16.pickle", "rb" ) )
    for key in dic:
        k = k+1
    print "Dist16", k
    k=0
    dic = pickle.load( open( "biodictdist32.pickle", "rb" ) )
    for key in dic:
        k = k+1
    print "Dist32", k    
    k=0
    dic = pickle.load( open( "biodictdist64.pickle", "rb" ) )
    for key in dic:
        k = k+1
    print "Dist64", k
