using Test

##############################################################################################

using Particles: AmortizedComputation, OfflineAC, AmortizedComputationTuple, remove_redundant_amortized

struct AC1 <: AmortizedComputation end
struct AC2 <: AmortizedComputation end
struct AC3 <: AmortizedComputation
    n::Int
end

function test_remove_redundant_amortized(amortized::AmortizedComputationTuple, reference::AmortizedComputationTuple)
    @test Set(remove_redundant_amortized(amortized)) == Set(reference)
end

function test_remove_redundant_amortized()
    let reference = (AC1(), AC2(), AC3(3), AC3(4), AC3(5))
        test_remove_redundant_amortized((AC1(), AC1(), AC2(), AC2(), AC2(), AC3(3), AC3(4), AC3(4), AC3(5)), reference)
        test_remove_redundant_amortized((AC2(), AC3(3), AC2(), AC3(4), AC1(), AC2(), AC3(4), AC3(5), AC1()), reference)
        test_remove_redundant_amortized((AC3(5), AC2(), AC3(4), AC2(), AC1(), AC2(), AC1(), AC3(4), AC3(3)), reference)
    end
    let reference = (OfflineAC(AC1()), AC2(), AC3(3), AC3(4), OfflineAC(AC3(5)))
        test_remove_redundant_amortized((AC1(), AC1(), OfflineAC(AC1()), OfflineAC(AC1()), AC2(), AC2(), AC2(), AC3(3), AC3(4), AC3(4), AC3(5), OfflineAC(AC3(5))), reference)
        test_remove_redundant_amortized((OfflineAC(AC3(5)), AC3(4), AC2(), AC3(3), OfflineAC(AC1()), AC2(), AC3(5), AC2(), AC3(4), AC1(), AC3(4), AC1(), OfflineAC(AC1()), AC2(), AC3(4)), reference)
    end
end

##############################################################################################

@testset "Summaries" begin
    @testset "AmortizedComputations" begin
        @testset "remove_redundant_amortized" test_remove_redundant_amortized()
    end
end
