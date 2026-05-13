#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

class Graph
{
    int V;
    vector<vector<int>> adj;

public:
    Graph(int vertices)
    {
        V = vertices;
        adj.resize(V);
    }

    void addEdge(int u, int v)
    {
        adj[u].push_back(v);
        adj[v].push_back(u); // undirected graph
    }

    // ---------- Sequential BFS ----------
    void bfsSequential(int start)
    {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        while(!q.empty())
        {
            int node = q.front();
            q.pop();

            cout << node << " ";

            for(int neighbor : adj[node])
            {
                if(!visited[neighbor])
                {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
    }

    // ---------- Parallel BFS ----------
    void bfsParallel(int start)
    {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        while(!q.empty())
        {
            int size = q.size();

            vector<int> currentLevel;

            while(size--)
            {
                currentLevel.push_back(q.front());
                q.pop();
            }

            #pragma omp parallel for
            for(int i = 0; i < currentLevel.size(); i++)
            {
                int node = currentLevel[i];

                #pragma omp critical
                cout << node << " ";

                for(int neighbor : adj[node])
                {
                    if(!visited[neighbor])
                    {
                        #pragma omp critical
                        {
                            if(!visited[neighbor])
                            {
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }
            }
        }
    }

    // ---------- Sequential DFS ----------
    void dfsSequentialUtil(int node, vector<bool>& visited)
    {
        visited[node] = true;

        cout << node << " ";

        for(int neighbor : adj[node])
        {
            if(!visited[neighbor])
                dfsSequentialUtil(neighbor, visited);
        }
    }

    void dfsSequential(int start)
    {
        vector<bool> visited(V, false);
        dfsSequentialUtil(start, visited);
    }

    // ---------- Parallel DFS ----------
    void dfsParallelUtil(int node, vector<bool>& visited)
    {
        visited[node] = true;

        #pragma omp critical
        cout << node << " ";

        for(int neighbor : adj[node])
        {
            if(!visited[neighbor])
            {
                #pragma omp task
                dfsParallelUtil(neighbor, visited);
            }
        }
    }

    void dfsParallel(int start)
    {
        vector<bool> visited(V, false);

        #pragma omp parallel
        {
            #pragma omp single
            dfsParallelUtil(start, visited);
        }
    }
};

int main()
{
    Graph g(7);

    g.addEdge(0,1);
    g.addEdge(0,2);
    g.addEdge(1,3);
    g.addEdge(1,4);
    g.addEdge(2,5);
    g.addEdge(2,6);

    // ---------- Sequential BFS ----------
    cout << "\nSequential BFS: ";

    double start = omp_get_wtime();

    g.bfsSequential(0);

    double end = omp_get_wtime();

    double seqBFS = end - start;

    // ---------- Parallel BFS ----------
    cout << "\nParallel BFS: ";

    start = omp_get_wtime();

    g.bfsParallel(0);

    end = omp_get_wtime();

    double parBFS = end - start;

    // ---------- Sequential DFS ----------
    cout << "\n\nSequential DFS: ";

    start = omp_get_wtime();

    g.dfsSequential(0);

    end = omp_get_wtime();

    double seqDFS = end - start;

    // ---------- Parallel DFS ----------
    cout << "\nParallel DFS: ";

    start = omp_get_wtime();

    g.dfsParallel(0);

    end = omp_get_wtime();

    double parDFS = end - start;

    // ---------- Performance ----------
    cout << "\n\n--- Performance ---";

    cout << "\nSequential BFS Time: " << seqBFS << " sec";
    cout << "\nParallel BFS Time: " << parBFS << " sec";
    cout << "\nBFS Speedup: " << seqBFS / parBFS << endl;

    cout << "\nSequential DFS Time: " << seqDFS << " sec";
    cout << "\nParallel DFS Time: " << parDFS << " sec";
    cout << "\nDFS Speedup: " << seqDFS / parDFS << endl;

    return 0;
}