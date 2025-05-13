#include <iostream>
#include <vector>
#include <cstdlib> // For rand()
#include <ctime>   // For srand()
#include <algorithm> // For std::max_element
#include <iomanip> // For printing the Q-table

using namespace std;

const int GRID_ROWS = 5;
const int GRID_COLS = 5;
const int NUM_STATES = GRID_ROWS * GRID_COLS; // 25 states
const int NUM_ACTIONS = 4; // 0:Up, 1:Down, 2:Left, 3:Right

// Action definitions
enum Action { UP, DOWN, LEFT, RIGHT };

// Cell types
enum CellType { START = 'S', N = 'N', MONSTER = 'M', GOAL = 'G' };

// Q-Learning Parameters
const double GAMMA = 0.9;         // Discount factor
const double EPSILON_START = 1.0; // Initial exploration rate
const double EPSILON_MIN = 0.01;  // Minimum exploration rate
const double EPSILON_DECAY = 0.995; // Decay rate for epsilon per episode
const int NUM_EPISODES = 10000;    // Total episodes for training
const int MAX_STEPS_PER_EPISODE = 100; // Prevent infinite loops in an episode

// Q-table: Q[state_index][action_index]
vector<vector<double>> q_table(NUM_STATES, vector<double>(NUM_ACTIONS, 0));

char grid[GRID_ROWS][GRID_COLS] = {
    START,   N,       N,       MONSTER, N,
    N,       N,       N,       N,       N,
    N,       N,       N,       N,       N,
    N,       MONSTER, N,       N,       N,
    N,       N,       N,       N,       GOAL
};
pair<int, int> start_state_index = {0, 0};

// Convert (row, col) to 1D state index
int to_state_index(int r, int c) {
    if (r >= 0 && r < GRID_ROWS && c >= 0 && c < GRID_COLS) {
        return r * GRID_COLS + c;
    }
    return -1; // Invalid index
}

// Convert 1D state index to (row, col)
pair<int, int> to_row_col(int state_index) {
    return {state_index / GRID_COLS, state_index % GRID_COLS};
}

// Get the reward for landing in a particular state_index
double get_reward(int state_index) {
    pair<int, int> row_col = to_row_col(state_index);
    int row = row_col.first, col = row_col.second;

    if (grid[row][col] == GOAL) return 100;  // Large positive reward for goal
    if (grid[row][col] == MONSTER) return -100; // Large negative reward for monster
    return -0.1; // Small negative reward for each step on normal ground (to get shorter paths)
}

// Determine the next state given current_state and action, considering slipping
int get_next_state(int current_state_index, Action action) {
    pair<int, int> current_pos = to_row_col(current_state_index);
    int r = current_pos.first;
    int c = current_pos.second;

    int dr = 0, dc = 0; // Change in row, col for the first step

    switch (action) {
        case UP:    dr = -1; break;
        case DOWN:  dr = 1;  break;
        case LEFT:  dc = -1; break;
        case RIGHT: dc = 1;  break;
    }

    // First step
    int next_r = r + dr;
    int next_c = c + dc;

    // Boundary check for the first step
    if (next_r < 0 || next_r >= GRID_ROWS || next_c < 0 || next_c >= GRID_COLS) {
        return current_state_index; // Hit a wall, stay in place
    }

    // Slipping: 50% chance to move one more step in the SAME direction
    if (static_cast<double>(rand()) / RAND_MAX < 0.5) { // 50% chance of slipping
        int slip_r = next_r + dr;
        int slip_c = next_c + dc;

        if (slip_r >= 0 && slip_r < GRID_ROWS && slip_c >= 0 && slip_c < GRID_COLS) {
            next_r = slip_r;
            next_c = slip_c;
        } else {
            // hit wall, stay the same pos 
        }
    }
    return to_state_index(next_r, next_c);
}


// Choose an action based on epsilon-greedy strategy
Action choose_action(int state_index, double epsilon) {
    if (static_cast<double>(rand()) / RAND_MAX < epsilon) {
        // Explore: choose a random action
        return  (Action)(rand() % NUM_ACTIONS);
    } else {
        // Exploit: choose the best known action
        double max_q = -1e9; // Initialize with a very small number
        Action best_action = UP; // Default action
        for (int i = 0; i < NUM_ACTIONS; ++i) {
            if (q_table[state_index][i] > max_q) {
                max_q = q_table[state_index][i];
                best_action = static_cast<Action>(i);
            }
        }
        return best_action;
    }
}

// Get the maximum Q-value for a given state (Max(Q(N(s,x), all actions)))
double get_max_q_value_for_state(int state_index) {
    if (state_index < 0 || state_index >= NUM_STATES) return 0.0; // Should not happen if next_state is not valid
    double max_q = -1e9; // Initialize with a very small number
    for (int i = 0; i < NUM_ACTIONS; ++i) {
         if (q_table[state_index][i] > max_q) {
            max_q = q_table[state_index][i];
        }
    }
    // If all Q-values for the next state are 0 (e.g., initial state or terminal state with no further actions),
    // this will return 0, which is correct as there's no future reward from that state.
    // However, for terminal states (Goal/Monster), their Q-values are effectively 0 as no more actions can be taken *from* them.
    // The reward R(s,a) leading to a terminal state is what matters.
    pair<int, int> row_col = to_row_col(state_index);
    int row = row_col.first,  col = row_col.second;
    if (grid[row][col] == GOAL || grid[row][col] == MONSTER) {
        return 0.0; // No future reward from a terminal state itself
    }

    return max_q;
}

// --- Main Q-Learning Logic ---
void run_q_learning() {
    srand(time(0)); // Seed random number generator
    double current_epsilon = EPSILON_START;

    for (int episode = 0; episode < NUM_EPISODES; ++episode) {
        int row = start_state_index.first;
        int col = start_state_index.second;
        int current_state = to_state_index(start_state_index.first, start_state_index.second);
        int steps = 0;


        while (steps < MAX_STEPS_PER_EPISODE) {
            pair<int, int> current_pos_coords = to_row_col(current_state);
            int r_curr = current_pos_coords.first; // Get current row
            int c_curr = current_pos_coords.second; // Get current col

            if (grid[r_curr][c_curr] == GOAL || grid[r_curr][c_curr] == MONSTER) {
                // ]cout << "Reached terminal state: " << grid[r_curr][c_curr] <<endl;
                break;
            }


            Action action_taken = choose_action(current_state, current_epsilon);
            int next_actual_state = get_next_state(current_state, action_taken);
            double reward = get_reward(next_actual_state); // Reward is for landing in next_actual_state

            // Q(s, x) = R(s, x) + y * Max(Q(N(s, x), all actions))
            // Here, R(s,x) is the reward obtained by taking action x from state s and landing in N(s,x)
            // So, reward 'reward' is R(current_state, action_taken) effectively.
            // Max(Q(N(s,x), all_actions)) is the max Q value of the 'next_actual_state'.
            double max_q_next_state = get_max_q_value_for_state(next_actual_state);

            // Update Q-value for the (current_state, action_taken) pair
            q_table[current_state][(int)(action_taken)] = reward + GAMMA * max_q_next_state;

            current_state = next_actual_state;
            steps++;
        }

        // Decay epsilon
        if (current_epsilon > EPSILON_MIN) {
            current_epsilon *= EPSILON_DECAY;
        }
        if ((episode + 1) % 1000 == 0) {
            cout << "Episode " << episode + 1 << "/" << NUM_EPISODES << " completed. Epsilon: " << current_epsilon << endl;
        }
    }
}

void print_q_table() {
    cout << "\n--- Q-Table ---" << endl;
    cout << fixed << setprecision(2);

    cout << setw(12) << "State";

    cout << setw(10) << "Up" << setw(10) << "Down" << setw(10) << "Left" << setw(10) << "Right\n";

    // Print each row of the table
    for (int s = 0; s < NUM_STATES; ++s) {
        cout << setw(12) << s;
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            cout << setw(10) << q_table[s][a];
        }
        cout << endl;
    }
}

void play_game() {
    cout << "\n--- Playing Game with Learned Policy ---" << endl;
    int current_state = to_state_index(start_state_index.first, start_state_index.second);
    int steps = 0;

    while (steps < MAX_STEPS_PER_EPISODE) {
        pair<int, int> pos = to_row_col(current_state);
        cout << "Step " << steps + 1 << ": At state (" << pos.first << "," << pos.second << ") which is '" << grid[pos.first][pos.second] << "'" << endl;

        if (grid[pos.first][pos.second] == GOAL) {  
            cout << "Goal reached!" << endl;
            break;
        }
        if (grid[pos.first][pos.second] == MONSTER) {
            cout << "Oops! Eaten by a monster!" << endl;
            break;
        }

        Action best_action = choose_action(current_state, 0.0);
        string action_str;
        switch(best_action){
            case UP: action_str="UP"; break;
            case DOWN: action_str="DOWN"; break;
            case LEFT: action_str="LEFT"; break;
            case RIGHT: action_str="RIGHT"; break;
        }
        cout << "  Choosing action: " << action_str << endl;


        int next_state = get_next_state(current_state, best_action);

        pair<int, int> next_pos = to_row_col(next_state);
        cout << "  Moved to state (" << next_pos.first << "," << next_pos.second << ") which is '" << grid[next_pos.first][next_pos.second] << "'" << endl;


        if (current_state == next_state && (best_action == UP && pos.first == 0 || best_action == DOWN && pos.first == GRID_ROWS -1 || best_action == LEFT && pos.second == 0 || best_action == RIGHT && pos.second == GRID_COLS -1)){
            cout << "Hit a wall and stayed ghaaad " << endl;
        }


        current_state = next_state;
        steps++;

        if (steps >= MAX_STEPS_PER_EPISODE) {
            cout << "Max steps reached. Did not find goal or monster." << endl;
        }
    }
}


int main() {

    run_q_learning();
    print_q_table();
    play_game();

    return 0;
}