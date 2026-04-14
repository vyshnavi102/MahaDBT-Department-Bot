import networkx as nx
import requests

G = nx.DiGraph()

STATUS_BUCKET = {
    6: "Approved",
    7: "Approved",
    3: "Rejected"
}

def fetch_data():
    query = """
    {
      applications(district: "Mumbai") {
        applicationno
        statusid
        districtname
      }
    }
    """

    response = requests.post(
        "http://127.0.0.1:5000/graphql",
        json={"query": query}
    )

    return response.json()["data"]["applications"]

def build_graph(data):
    for row in data:
        app = row["applicationno"]
        status = row["statusid"]

        # Add application node
        G.add_node(app, type="Application", district=row["districtname"], statusid=status)

        # Add status node
        status_node = f"status_{status}"
        G.add_node(status_node, type="Status", bucket=STATUS_BUCKET.get(status, "Unknown"))

        # Connect them
        G.add_edge(app, status_node)

def count_approved(district):
    count = 0

    for node in G.nodes:
        data = G.nodes[node]

        if data.get("type") == "Application":
            if data.get("district") == district:
                if data.get("statusid") in [6,7]:
                    count += 1

    return count

if __name__ == "__main__":
    data = fetch_data()
    build_graph(data)

    result = count_approved("Mumbai")
    print("Approved applications in Mumbai:", result)
    
