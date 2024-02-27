from flask import Flask, Response,render_template, request, jsonify
from flask_cors import CORS, cross_origin
import rl
import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/run_rl',methods=['POST'])
@cross_origin()
def run_rl():
    def tup(l):
            return [(x,y) for [x,y] in l]
    try:
        d = request.get_json()
        
        # print(d["data"])
        goal_sets=[]
        start_positions=[]
        # #process json data to get the below format data
        for i in d["data"]:
            l=[]
            for j in i["set"]:
                x,y= j.split(",")
                x=int(x)
                y=int(y)
                l.append((x,y))
            goal_sets.append(l)
            x,y=i["start"].split(",")
            x,y=int(x),int(y)
            start_positions.append((x,y))
        grid_size=int(d["n"])
        # print(data)
        # RL function call from notebook
        output = rl.run_rl(start_positions,goal_sets,[grid_size,grid_size])#
        # output=rl.run_rl(start_positions=[(0,0),(4,0)],goal_sets=[[(2, 1), (2, 2)], [(4, 3), (3, 4)]],grid_size=[5,5])
        json_output=json.loads(output)
        for i, item in enumerate(d["data"]):
            item["path"] = json_output[i].get("path", None)
            # print(item["path"])
        # for i in json_output:
        #      print(i["path"])

        # Return a JSON response with modified "data"
        # return jsonify({"data": d["data"]})
        return d
        

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
