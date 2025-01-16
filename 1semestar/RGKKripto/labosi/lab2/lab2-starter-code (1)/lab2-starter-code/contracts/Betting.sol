// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./BoxOracle.sol";

contract Betting {

    struct Player {
        uint8 id;
        string name;
        uint totalBetAmount;
        uint currCoef; 
    }
    struct Bet {
        address bettor;
        uint amount;
        uint player_id;
        uint betCoef;
    }

    address private betMaker;
    BoxOracle public oracle;
    uint public minBetAmount;
    uint public maxBetAmount;
    uint public totalBetAmount;
    uint public thresholdAmount;

    Bet[] private bets;
    Player public player_1;
    Player public player_2;

    bool private suspended = false;
    mapping (address => uint) public balances;
    
    constructor(
        address _betMaker,
        string memory _player_1,
        string memory _player_2,
        uint _minBetAmount,
        uint _maxBetAmount,
        uint _thresholdAmount,
        BoxOracle _oracle
    ) {
        betMaker = (_betMaker == address(0) ? msg.sender : _betMaker);
        player_1 = Player(1, _player_1, 0, 200);
        player_2 = Player(2, _player_2, 0, 200);
        minBetAmount = _minBetAmount;
        maxBetAmount = _maxBetAmount;
        thresholdAmount = _thresholdAmount;
        oracle = _oracle;

        totalBetAmount = 0;
    }

    receive() external payable {}

    fallback() external payable {}

    function adaptCoef() private{
        player_1.currCoef = (totalBetAmount*100) / player_1.totalBetAmount;
        player_2.currCoef = (totalBetAmount*100) / player_2.totalBetAmount;
    }
    
    function makeBet(uint8 _playerId) public payable {
        //TODO Your code here
        //revert("Not yet implemented");
        require(!suspended, "Betting is suspended");
        require(msg.sender != betMaker, "Owner cannot bet!");
        require(msg.value >= minBetAmount, "You have to bet more than that!");
        require(msg.value <= maxBetAmount, "You have to bet less than that!");
        require(_playerId == 1 || _playerId == 2, "Only IDs 1 or 2 are accepted");
        require(oracle.getWinner() == 0, "Winner shouldn't be decided yet!");

        if (_playerId == 1) {
            player_1.totalBetAmount += msg.value;
        } else {
            player_2.totalBetAmount += msg.value;
        }

        totalBetAmount += msg.value;

        uint currentCoef = _playerId == 1 ? player_1.currCoef : player_2.currCoef;
        bets.push(Bet(msg.sender, msg.value, _playerId, currentCoef));

        if(totalBetAmount > thresholdAmount && totalBetAmount-msg.value < thresholdAmount){
            if(player_1.totalBetAmount == 0 || player_2.totalBetAmount == 0){
                suspended = true;
                return;
            }
            adaptCoef();
        } else if (totalBetAmount > thresholdAmount){
            adaptCoef();
        }
    }


    function claimSuspendedBets() public {
        //TODO Your code here
        //revert("Not yet implemented");
        require(suspended, "Betting isn't suspended!");
        uint refund = 0;

        for (uint i = 0; i < bets.length; i++){
            if (bets[i].bettor == msg.sender){
                refund += bets[i].amount;
                bets[i].amount = 0;     // Deleting from contract memory
            }
        }

        require(refund > 0, "Nothing to claim");
        payable(msg.sender).transfer(refund); 
    }

    function claimWinningBets() public {
        uint8 winner = oracle.getWinner();
        //emit ClaimAttempt(msg.sender, winner, suspended, 0);
        require(winner == 1 || winner == 2, "Match not decided yet!");
        require(!suspended, "This betting was suspended");
        //emit ClaimAttempt(msg.sender, winner, suspended, 0);
        uint winnings = 0;

        for (uint i = 0; i < bets.length; i++) {
            if (bets[i].bettor == msg.sender && bets[i].player_id == winner && bets[i].amount > 0) {
                winnings += (bets[i].amount * bets[i].betCoef) / 100;
                bets[i].amount = 0;
            }
        }
        bool sent = payable(msg.sender).send(winnings);
        require(sent, "Failed to send Ether");
    }

    event Error(
        string Kak,
        uint amount
    );

    function claimLosingBets() public {
        // TODO Your code here
        //revert("Not yet implemented");
        uint8 winner = oracle.getWinner();
        require(winner == 1 || winner == 2, "Match not decided yet!");
        require(!suspended, "This betting was suspended");
        require(msg.sender == betMaker, "Only bet maker can claim loosing funds");
        
        uint payday = 0;

        bool allPaid = true;

        for (uint i = 0; i < bets.length; i++){
            if (bets[i].player_id == winner && bets[i].amount > 0){
                allPaid = false;
            }

            if(bets[i].player_id != winner){
                payday += bets[i].amount;
                bets[i].amount = 0;
            }
        }
        require(allPaid, "There are still people who haven't collected their winnings!");
        emit Error("Svi pokupili", payday);

        bool sent = payable(betMaker).send(payday);
        require(sent, "Failed to send Ether");
    }
}