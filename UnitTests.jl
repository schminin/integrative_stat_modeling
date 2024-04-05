
######################################################################
# Testing Area for the CaseCountDistribution and CaseCountParameterMapping
######################################################################

include("CaseCountDistributions.jl")

# test function for initial states
function check_initial_state(state, m_Λ)
    Y = state[1:m_Λ+1]
    A = state[m_Λ+2:2*(m_Λ+1)]
    A_sum = state[2*m_Λ+3:3*m_Λ+1]
    R = state[end]
    if any(A.>Y)
        println("A cannot be larger than Y")
        println("Error at: $(.!(A.<=Y))")
    end
    if any(A_sum.>Y[2:end-1])
        println("A_sum cannot be larger than Y")
        println("Error at: $(.!(A_sum.<=Y[2:end-1]))")
    end
    if any(A[2:end-1].>A_sum)
        println("A_{t-i, i} cannot be larger than A_sum_{t}")
        println("Error at: $(.!(A[2:end-1].<=A_sum))")
    end
    if R < 0
        println("R cannot be negative")
    end
end

old_state_2 = SVector(8.0, 7.0, 6.0, 5.0, 2.0, 3.0, 4.0, 1)
check_initial_state(old_state_2, 2)


old_state_4 = SVector(4.0, 8.0, 7.0, 15.0, 3.0, 2.0, 3.0, 6.0, 2.0, 2.0, 7.0, 6.0, 2.0, 1.0)
check_initial_state(old_state_4, 4)


function test_distribution_update(old_state)
    cc = CaseCountDistribution(old_state, SVector(2.0, 2.0))
    
    old_Y = old_state[1:cc.m_Λ+1]
    old_A = old_state[cc.m_Λ+2:2*(cc.m_Λ+1)]
    old_A_sum = old_state[2*cc.m_Λ+3:3*cc.m_Λ+1]
    old_R = old_state[end]

    new_state = rand(cc)
    new_Y = new_state[1:cc.m_Λ+1]
    new_A = new_state[cc.m_Λ+2:2*(cc.m_Λ+1)]
    new_A_sum = new_state[2*cc.m_Λ+3:3*cc.m_Λ+1]
    new_R = new_state[end]

    # test 1: Update Y is likely correct
    if ! (new_Y[2:end] == old_Y[1:end-1])
        println("Error in Y update: Values are not shifted correctly")
    end
    if ! (isinteger(new_Y[1]))
        println("Error in Y update: Y[1] is not an integer")
    end
    if (new_Y[2:end] == old_Y[1:end-1]) & isinteger(new_Y[1])
        println("Update of Y is correct \n")
    end

    # test 2: Update A is likely correct
    # subtest a): A[1] is smaller than Y[1]
    if ! (new_A[1] <= new_Y[1])
        print("Error in A update: A[1] is larger than Y[1]")
    end
    # subtest b): sum of old and new A of one timepoint of infection are smaller than corresponding Y
    if !all(new_A[2:end] + old_A[1:end-1] .<= new_Y[2:end])
        println("Error in A update: sum of two A's is larger than corresponding Y")
        println("Error (1) at: $(.!(new_A[2:end] + old_A[1:end-1] .<= old_Y[1:end-1]))")
    end
    println("Update of A looks good")
    
    # test 3: Update of A_sum is likely correct
    if !(new_A_sum[1] == old_A[1] + new_A[2])
        print("Error in A_sum update: A_sum[1] is not correct")
    end
    println("Update of sum A looks good")

end

# perform 10 tests
[test_distribution_update(old_state_2) for i in 1:10]

[test_distribution_update(old_state_4) for i in 1:10]



