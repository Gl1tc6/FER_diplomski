// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Timer.sol";

/// This contract represents most simple crowdfunding campaign.
/// This contract does not protects investors from not receiving goods
/// they were promised from crowdfunding owner. This kind of contract
/// might be suitable for campaigns that does not promise anything to the
/// investors except that they will start working on some project.
/// (e.g. almost all blockchain spinoffs.)
contract Crowdfunding {

    address private owner;

    Timer private timer;

    uint256 public goal;

    uint256 public endTimestamp;

    mapping (address => uint256) public investments;

    constructor(
        address _owner,
        Timer _timer,
        uint256 _goal,
        uint256 _endTimestamp
    ) {
        owner = (_owner == address(0) ? msg.sender : _owner);
        timer = _timer; // Not checking if this is correctly injected.
        goal = _goal;
        endTimestamp = _endTimestamp;
    }
    
    event Investment(address indexed investor, uint256 amount); 

    function invest() public payable {
        // TODO Your code here
        //revert("Not yet implemented");
        require(timer.getTime() <= endTimestamp, "Campaign has ended!");
        require(msg.value > 0, "Value has to be positive float value!");

        if (investments[msg.sender] == 0){
            investments[msg.sender] = msg.value;
        }
        else{
            investments[msg.sender] += msg.value;
        }
        //emit Investment(msg.sender, msg.value);
    }

    function claimFunds() public {
        // TODO Your code here
        //revert("Not yet implemented");
        require(msg.sender == owner, "Only owner can claim funds");
        uint256 raised = address(this).balance;
        require((timer.getTime() > endTimestamp) && (raised >= goal), "Goal not reached or campaign hasn't ended!");
        
        (bool sent, ) = owner.call{value: raised}("");
        require(sent, "Failed to send funds to owner");
    }

    function refund() public {
        // TODO Your code here
        // revert("Not yet implemented");
        uint256 raised = address(this).balance;

        require((timer.getTime() > endTimestamp) && (raised < goal), "Goal was reached or campaign hasn't ended! Refund is not possible!");
        
        uint256 invested = investments[msg.sender];

        require(invested > 0, "Nothing to refund...");
        investments[msg.sender] = 0;
        
        (bool sent, ) = msg.sender.call{value: invested}("");
        require(sent, "Failed to refund");
    }
    
}