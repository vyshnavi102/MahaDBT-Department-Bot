from flask import Flask, request, jsonify
import graphene


# Mock data
data = [
    {"applicationno": "A1", "statusid": 6, "districtname": "Mumbai"},
    {"applicationno": "A2", "statusid": 3, "districtname": "Mumbai"},
    {"applicationno": "A3", "statusid": 7, "districtname": "Pune"},
]

class Application(graphene.ObjectType):
    applicationno = graphene.String()
    statusid = graphene.Int()
    districtname = graphene.String()

class Query(graphene.ObjectType):
    applications = graphene.List(Application, district=graphene.String())

    def resolve_applications(root, info, district=None):
        if district:
            return [a for a in data if a["districtname"] == district]
        return data

schema = graphene.Schema(query=Query)

app = Flask(__name__)

@app.route("/graphql", methods=["POST"])
def graphql_api():
    data_req = request.get_json()
    result = schema.execute(data_req.get("query"))
    return jsonify({"data": result.data})

if __name__ == "__main__":
    app.run(debug=True)
